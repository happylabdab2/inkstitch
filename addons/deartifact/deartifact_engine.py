"""
De-artifacting engine for bitmap-to-SVG conversion.

Pure numpy implementation (runs on Inkscape's bundled Python, no SciPy/OpenCV).

Pipeline
    1. Analysis            noise estimate, 8x8 blocking, unique colours
    2. Restoration         bilateral filter in CIELAB + noise-adaptive extra pass
    3. Segmentation        SLIC superpixels, split into flat zones (thin lines survive)
    4. Region graph (RAG)  hierarchical merging, CIEDE2000 + edge + variance cost
    5. Transitions         antialiasing strips are split between their two parents
    6. Islands             logistic artifact score, topology-aware absorption
    7. Palette             region-level weighted k-means / leader clustering,
                           weighted-median canonical colours, protected colours
    8. Tracing             planar boundary graph -> shared chains -> Bezier fitting
                           (each boundary is fitted once, so there are no gaps)
"""

import heapq
import math

import numpy as np

# --------------------------------------------------------------------------
# Colour science
# --------------------------------------------------------------------------

_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS = 216.0 / 24389.0
_KAP = 24389.0 / 27.0


def srgb_to_lab(rgb):
    """rgb: float array in 0..1, shape (...,3) -> CIELAB (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAP * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_srgb(lab):
    """CIELAB -> sRGB float 0..1 (clamped to gamut)."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xr = np.where(fx**3 > _EPS, fx**3, (116.0 * fx - 16.0) / _KAP)
    yr = np.where(L > _KAP * _EPS, fy**3, L / _KAP)
    zr = np.where(fz**3 > _EPS, fz**3, (116.0 * fz - 16.0) / _KAP)
    xyz = np.stack([xr, yr, zr], -1) * _WHITE
    lin = np.clip(xyz @ _M_XYZ2RGB.T, 0.0, 1.0)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


def lab_to_hex(lab):
    r, g, b = (np.round(lab_to_srgb(lab) * 255)).astype(int)
    return "#%02x%02x%02x" % (r, g, b)


def hex_to_lab(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError("bad colour '%s'" % h)
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return srgb_to_lab(rgb)


_P25_7 = 25.0**7


def de2000(c1, c2):
    """Scalar CIEDE2000 (fast path for the region graph)."""
    L1, a1, b1 = c1
    L2, a2, b2 = c2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cb7 = ((C1 + C2) * 0.5) ** 7
    G = 0.5 * (1.0 - math.sqrt(Cb7 / (Cb7 + _P25_7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    cc = C1p * C2p
    if cc == 0.0:
        dhp = 0.0
        hbp = h1p + h2p
    else:
        dhp = h2p - h1p
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0
        s = h1p + h2p
        if abs(h1p - h2p) <= 180.0:
            hbp = s * 0.5
        elif s < 360.0:
            hbp = (s + 360.0) * 0.5
        else:
            hbp = (s - 360.0) * 0.5
    dHp = 2.0 * math.sqrt(cc) * math.sin(math.radians(dhp) * 0.5)
    Lbp = (L1 + L2) * 0.5
    Cbp = (C1p + C2p) * 0.5
    T = (
        1.0
        - 0.17 * math.cos(math.radians(hbp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hbp))
        + 0.32 * math.cos(math.radians(3.0 * hbp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hbp - 63.0))
    )
    dth = 30.0 * math.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp**7
    RC = 2.0 * math.sqrt(Cbp7 / (Cbp7 + _P25_7))
    l50 = (Lbp - 50.0) ** 2
    SL = 1.0 + 0.015 * l50 / math.sqrt(20.0 + l50)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2.0 * dth)) * RC
    tl, tc, th = dLp / SL, dCp / SC, dHp / SH
    return math.sqrt(max(0.0, tl * tl + tc * tc + th * th + RT * tc * th))


def pde(c1, c2):
    """
    Decision metric: CIEDE2000 with chroma damped in dark tones. Chroma
    discrimination collapses near black, but CIELAB/CIEDE2000 still report
    large differences there, so near-black JPEG chroma noise would otherwise
    split one black shape into several 'colours'.
    """
    Lm = 0.5 * (c1[0] + c2[0])
    if Lm >= 25.0:
        return de2000(c1, c2)
    f = max(0.3, Lm / 25.0)
    return de2000((c1[0], c1[1] * f, c1[2] * f), (c2[0], c2[1] * f, c2[2] * f))


def pde_vec(lab1, lab2):
    lab1 = np.asarray(lab1, float)
    lab2 = np.asarray(lab2, float)
    Lm = 0.5 * (lab1[..., 0] + lab2[..., 0])
    f = np.clip(Lm / 25.0, 0.3, 1.0)[..., None]
    s = np.concatenate([np.ones(f.shape), f, f], -1)
    return de2000_vec(lab1 * s, lab2 * s)


def de2000_vec(lab1, lab2):
    """Vectorised CIEDE2000; broadcasts (...,3) arrays."""
    lab1 = np.asarray(lab1, float)
    lab2 = np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb7 = ((C1 + C2) * 0.5) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + _P25_7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    cc = C1p * C2p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(cc == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(cc) * np.sin(np.radians(dhp) / 2)
    s = h1p + h2p
    hbp = np.where(
        np.abs(h1p - h2p) <= 180, s / 2, np.where(s < 360, (s + 360) / 2, (s - 360) / 2)
    )
    hbp = np.where(cc == 0, s, hbp)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    T = (
        1
        - 0.17 * np.cos(np.radians(hbp - 30))
        + 0.24 * np.cos(np.radians(2 * hbp))
        + 0.32 * np.cos(np.radians(3 * hbp + 6))
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    )
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cbp**7 / (Cbp**7 + _P25_7))
    l50 = (Lbp - 50) ** 2
    SL = 1 + 0.015 * l50 / np.sqrt(20 + l50)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dth)) * RC
    tl, tc, th = dLp / SL, dCp / SC, dHp / SH
    return np.sqrt(np.maximum(0, tl**2 + tc**2 + th**2 + RT * tc * th))


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

PROFILES = {
    #               tol  island_sens edge_pres smooth_iters
    "conservative": (5.0, 2, 0.90, 1),
    "balanced": (8.0, 4, 0.75, 2),
    "aggressive": (13.0, 7, 0.50, 3),
}


class Config:
    def __init__(self, **kw):
        self.profile = "balanced"  # or 'custom'
        self.target_colors = 0  # 0 = automatic
        self.tolerance = 8.0  # Delta-E00 merge tolerance
        self.island_sensitivity = 4  # 0..10
        self.island_max_area = 0  # px, 0 = auto
        self.edge_preservation = 0.75  # 0..1
        self.protect_holes = True
        self.smooth_iterations = 2
        self.superpixel_size = 0  # px, 0 = auto
        self.protected_colors = []  # list of hex strings
        self.alpha_threshold = 128
        self.max_pixels = 2_000_000
        # tracing
        self.trace_tolerance = 1.0
        self.corner_angle = 55.0
        self.smoothness = 1.0
        self.__dict__.update(kw)
        if self.profile in PROFILES:
            (
                self.tolerance,
                self.island_sensitivity,
                self.edge_preservation,
                self.smooth_iterations,
            ) = PROFILES[self.profile]


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------


def _box_mean(x, r):
    """Mean over a (2r+1)^2 window via integral image; x is (H,W[,C])."""
    pad = [(r + 1, r), (r + 1, r)] + [(0, 0)] * (x.ndim - 2)
    ii = np.pad(x, pad, mode="edge").cumsum(0).cumsum(1)
    H, W = x.shape[:2]
    k = 2 * r + 1
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return s[:H, :W] / (k * k)


def connected_components(link_h, link_v, valid=None):
    """
    Label connected pixels. link_h[y,x] links (y,x)-(y,x+1), link_v[y,x] links
    (y,x)-(y+1,x). Pixels where valid is False get label -1.
    Vectorised union-find (hooking + pointer jumping).
    """
    H = link_v.shape[0] + 1
    W = link_h.shape[1] + 1
    n = H * W
    idx = np.arange(n, dtype=np.int64).reshape(H, W)
    ea = np.concatenate([idx[:, :-1][link_h], idx[:-1, :][link_v]])
    eb = np.concatenate([idx[:, 1:][link_h], idx[1:, :][link_v]])
    parent = np.arange(n, dtype=np.int64)
    while ea.size:
        pa = parent[ea]
        pb = parent[eb]
        keep = pa != pb
        if not keep.any():
            break
        ea, eb, pa, pb = ea[keep], eb[keep], pa[keep], pb[keep]
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        np.minimum.at(parent, hi, lo)
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent = nxt
    labels = parent.reshape(H, W)
    if valid is not None:
        labels = np.where(valid, labels, -1)
    return compact_labels(labels)


def compact_labels(labels):
    out = np.full(labels.shape, -1, dtype=np.int64)
    m = labels >= 0
    if m.any():
        _, inv = np.unique(labels[m], return_inverse=True)
        out[m] = inv
    return out


def relabel_connected(labels):
    """Split every label into its 4-connected components."""
    lh = (labels[:, :-1] == labels[:, 1:]) & (labels[:, :-1] >= 0)
    lv = (labels[:-1, :] == labels[1:, :]) & (labels[:-1, :] >= 0)
    return connected_components(lh, lv, labels >= 0)


# --------------------------------------------------------------------------
# 1. Analysis
# --------------------------------------------------------------------------


def analyse(rgb8, lab, valid):
    H, W = lab.shape[:2]
    info = {"width": W, "height": H}
    px = rgb8[valid]
    if px.size:
        packed = (px[:, 0].astype(np.int64) << 16) | (px[:, 1].astype(np.int64) << 8) | px[:, 2]
        info["unique_colors"] = int(np.unique(packed).size)
    else:
        info["unique_colors"] = 0

    # Noise (Immerkaer Laplacian, robust percentile so edges do not dominate)
    sig2 = 0.0
    if H > 3 and W > 3:
        for c in range(3):
            x = lab[..., c]
            conv = (
                x[:-2, :-2] - 2 * x[:-2, 1:-1] + x[:-2, 2:]
                - 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] - 2 * x[1:-1, 2:]
                + x[2:, :-2] - 2 * x[2:, 1:-1] + x[2:, 2:]
            )
            a = np.abs(conv[valid[1:-1, 1:-1]])
            if a.size:
                s = np.percentile(a, 75) / (1.15 * 6.0)
                sig2 += s * s
    info["noise_estimate"] = float(math.sqrt(sig2))
    info["noise"] = max(1.0, info["noise_estimate"])

    # JPEG 8x8 blocking: gradient energy on block boundaries vs elsewhere
    block = 1.0
    if W >= 32 and H >= 32:
        dx = np.abs(np.diff(lab[..., 0], axis=1))
        dy = np.abs(np.diff(lab[..., 0], axis=0))
        cx = np.arange(W - 1) % 8 == 7
        cy = np.arange(H - 1) % 8 == 7
        on = dx[:, cx].mean() + dy[cy, :].mean()
        off = dx[:, ~cx].mean() + dy[~cy, :].mean()
        block = float(on / off) if off > 1e-9 else 1.0
    info["blockiness"] = block
    info["jpeg_like"] = block > 1.12

    # Chroma subsampling: chroma has much less fine detail than luma
    if W > 8 and H > 8:
        def hf(ch):
            return np.abs(np.diff(ch, axis=1)).mean() + 1e-9
        ratio = (hf(lab[..., 1]) + hf(lab[..., 2])) / (2 * hf(lab[..., 0]))
        info["chroma_detail_ratio"] = float(ratio)
    return info


# --------------------------------------------------------------------------
# 2. Restoration
# --------------------------------------------------------------------------


def bilateral(lab, radius, sigma_s, sigma_r, iterations):
    out = lab.astype(np.float32)
    H, W, _ = out.shape
    r = radius
    inv_r = 1.0 / (2.0 * sigma_r * sigma_r)
    for _ in range(iterations):
        pad = np.pad(out, ((r, r), (r, r), (0, 0)), mode="edge")
        num = np.zeros_like(out)
        den = np.zeros((H, W), np.float32)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ws = math.exp(-(dy * dy + dx * dx) / (2.0 * sigma_s * sigma_s))
                sh = pad[r + dy : r + dy + H, r + dx : r + dx + W]
                diff = sh - out
                d2 = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
                w = np.float32(ws) * np.exp(-d2 * np.float32(inv_r))
                num += w[..., None] * sh
                den += w
        out = num / den[..., None]
    return out


def restore_chroma(lab, radius, sigma_L, sigma_c):
    """
    Joint bilateral on the a/b channels guided by lightness. Undoes chroma
    bleed from 4:2:0 subsampling: a pixel takes its chroma from neighbours
    of similar L. A loose chroma kernel keeps genuine isoluminant edges.
    """
    out = lab.astype(np.float32).copy()
    H, W, _ = out.shape
    r = radius
    pad = np.pad(lab, ((r, r), (r, r), (0, 0)), mode="edge").astype(np.float32)
    L0, A0, B0 = (np.ascontiguousarray(out[..., c]) for c in range(3))
    numa = np.zeros((H, W), np.float32)
    numb = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    kL = np.float32(1.0 / (2 * sigma_L * sigma_L))
    kc = np.float32(1.0 / (2 * sigma_c * sigma_c))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r:
                continue
            sh = pad[r + dy : r + dy + H, r + dx : r + dx + W]
            dL = sh[..., 0] - L0
            da = sh[..., 1] - A0
            db = sh[..., 2] - B0
            w = np.exp(-(dL * dL) * kL - (da * da + db * db) * kc)
            numa += w * sh[..., 1]
            numb += w * sh[..., 2]
            den += w
    out[..., 1] = numa / den
    out[..., 2] = numb / den
    return out


def gradient_magnitude(lab):
    p = np.pad(lab, ((1, 1), (1, 1), (0, 0)), mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.sqrt((gx**2 + gy**2).sum(-1)) / 8.0


def restore(lab, info, cfg):
    """Edge-aware restoration. Returns (restored_lab, noise_mask)."""
    noise = info["noise"]
    sigma_r = max(3.0, 2.5 * noise, 0.6 * cfg.tolerance)
    iters = cfg.smooth_iterations + (1 if info.get("jpeg_like") else 0)
    if iters <= 0:
        return lab.astype(np.float32), np.zeros(lab.shape[:2], bool)
    out = lab
    for _ in range(2 if info.get("jpeg_like") or iters > 1 else 1):
        out = restore_chroma(out, 3 + min(iters, 2), 2.0 + noise, 2.0 * cfg.tolerance + 4.0)
    out = bilateral(out, 2, 1.6, sigma_r, iters)

    # Local variance vs edge strength (high variance + low edge = noise)
    mean = _box_mean(lab, 1)
    var = np.maximum(_box_mean((lab**2).sum(-1), 1) - (mean**2).sum(-1), 0)
    edge = gradient_magnitude(out)
    thr_v = max(4.0, (2.0 * noise) ** 2)
    thr_e = max(3.0, 1.2 * cfg.tolerance)
    noise_mask = (var > thr_v) & (edge < thr_e)
    if noise_mask.any():
        strong = bilateral(out, 2, 2.0, sigma_r * 1.8, 1)
        out = np.where(noise_mask[..., None], strong, out)
    return out, noise_mask


# --------------------------------------------------------------------------
# 3. SLIC + flat-zone over-segmentation
# --------------------------------------------------------------------------


def slic(lab, step, compactness=10.0, iterations=5, valid=None):
    H, W, _ = lab.shape
    step = max(4, int(step))
    ny = max(1, int(math.ceil(H / step)))
    nx = max(1, int(math.ceil(W / step)))
    yy, xx = np.mgrid[0:H, 0:W]
    cy = (np.arange(ny) * step + step / 2.0).clip(0, H - 1)
    cx = (np.arange(nx) * step + step / 2.0).clip(0, W - 1)
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    CY = CY.ravel().astype(np.float64)
    CX = CX.ravel().astype(np.float64)
    CL = lab[CY.astype(int), CX.astype(int)].astype(np.float64)
    iy = np.minimum(yy // step, ny - 1)
    ix = np.minimum(xx // step, nx - 1)
    wspace = (compactness / step) ** 2
    K = ny * nx
    flat = lab.reshape(-1, 3)
    best = np.zeros((H, W), np.int64)
    for _ in range(iterations):
        best_d = np.full((H, W), np.inf, np.float32)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                k = np.clip(iy + oy, 0, ny - 1) * nx + np.clip(ix + ox, 0, nx - 1)
                diff = lab - CL[k].astype(np.float32)
                dc = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
                ds = (yy - CY[k]) ** 2 + (xx - CX[k]) ** 2
                d = dc + wspace * ds
                upd = d < best_d
                best_d[upd] = d[upd]
                best[upd] = k[upd]
        b = best.ravel()
        cnt = np.bincount(b, minlength=K).astype(np.float64)
        ok = cnt > 0
        for c in range(3):
            s = np.bincount(b, weights=flat[:, c], minlength=K)
            CL[ok, c] = s[ok] / cnt[ok]
        CY[ok] = np.bincount(b, weights=yy.ravel(), minlength=K)[ok] / cnt[ok]
        CX[ok] = np.bincount(b, weights=xx.ravel(), minlength=K)[ok] / cnt[ok]
    return best


def initial_segmentation(lab, valid, cfg, info):
    H, W = lab.shape[:2]
    step = cfg.superpixel_size or int(np.clip(math.sqrt(H * W / 1500.0), 6, 24))
    sp = slic(lab, step, compactness=8.0)
    link_t = max(1.5, 0.45 * cfg.tolerance, 1.5 * info["noise"])
    d_h = np.sqrt(((lab[:, 1:] - lab[:, :-1]) ** 2).sum(-1))
    d_v = np.sqrt(((lab[1:, :] - lab[:-1, :]) ** 2).sum(-1))
    lh = (sp[:, 1:] == sp[:, :-1]) & (d_h < link_t) & valid[:, 1:] & valid[:, :-1]
    lv = (sp[1:, :] == sp[:-1, :]) & (d_v < link_t) & valid[1:, :] & valid[:-1, :]
    return connected_components(lh, lv, valid), step


# --------------------------------------------------------------------------
# 4. Region adjacency graph
# --------------------------------------------------------------------------


class RegionGraph:
    """
    Nodes = regions of a label map. Each node keeps pixel count, Lab sum,
    sum of squared norms, perimeter and border length. Each edge keeps the
    shared boundary length and the summed raw contrast across it.
    """

    def __init__(self, labels, lab, contrast_h, contrast_v, noise, cfg, fixed_colors=None):
        self.cfg = cfg
        self.noise = noise
        n = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
        self.n = n
        flat = labels.ravel()
        m = flat >= 0
        fl = flat[m]
        lf = lab.reshape(-1, 3)[m].astype(np.float64)
        self.cnt = np.bincount(fl, minlength=n).astype(float).tolist()
        sums = [np.bincount(fl, weights=lf[:, c], minlength=n) for c in range(3)]
        self.sum = np.stack(sums, 1).tolist()
        self.sq = np.bincount(fl, weights=(lf**2).sum(1), minlength=n).tolist()
        self.parent = list(range(n))
        self.protected = [None] * n  # Lab tuple of protected colour or None
        self.fixed = fixed_colors  # optional per-node colour override (Lab)

        # adjacency
        a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
        b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
        c = np.concatenate([contrast_h.ravel(), contrast_v.ravel()])
        diff = a != b
        a, b, c = a[diff], b[diff], c[diff]
        # perimeter: every crack edge with a different label, incl. transparent
        per = np.zeros(n)
        for side in (a, b):
            s = side[side >= 0]
            per += np.bincount(s, minlength=n)
        both = (a >= 0) & (b >= 0)
        a, b, c = a[both], b[both], c[both]
        # image border
        border = np.zeros(n)
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            e = edge[edge >= 0]
            border += np.bincount(e, minlength=n)
        self.perim = (per + border).tolist()
        self.border = border.tolist()
        self.nb = [dict() for _ in range(n)]
        if a.size:
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            key = lo * n + hi
            uk, inv = np.unique(key, return_inverse=True)
            blen = np.bincount(inv).astype(float)
            bsum = np.bincount(inv, weights=c)
            for k, l, s in zip(uk.tolist(), blen.tolist(), bsum.tolist()):
                i, j = divmod(k, n)
                e = [l, s]
                self.nb[i][j] = e
                self.nb[j][i] = e
        # transparency adjacency marks regions as touching the "border"
        if (labels < 0).any():
            t = labels < 0
            touch = np.zeros_like(t)
            touch[:, :-1] |= t[:, 1:]
            touch[:, 1:] |= t[:, :-1]
            touch[:-1, :] |= t[1:, :]
            touch[1:, :] |= t[:-1, :]
            tl = labels[touch & (labels >= 0)]
            tb = np.bincount(tl, minlength=n)
            self.trans = tb.astype(float).tolist()
            for i in np.nonzero(tb)[0].tolist():
                self.border[i] += float(tb[i])
        else:
            self.trans = [0.0] * n

    # -- basic ops ----------------------------------------------------------
    def find(self, i):
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def roots(self):
        return [i for i in range(self.n) if self.parent[i] == i]

    def mean(self, i):
        if self.protected[i] is not None:
            return self.protected[i]
        if self.fixed is not None:
            return self.fixed[i]
        c = self.cnt[i]
        s = self.sum[i]
        return (s[0] / c, s[1] / c, s[2] / c)

    def raw_mean(self, i):
        c = self.cnt[i]
        s = self.sum[i]
        return (s[0] / c, s[1] / c, s[2] / c)

    def var(self, i):
        c = self.cnt[i]
        m = self.raw_mean(i)
        return max(0.0, self.sq[i] / c - (m[0] ** 2 + m[1] ** 2 + m[2] ** 2))

    def merge(self, a, b):
        """Merge root b into root a (or vice-versa, larger survives). Returns root."""
        if a == b:
            return a
        if self.cnt[a] < self.cnt[b]:
            a, b = b, a
        shared = self.nb[a].get(b, [0.0, 0.0])[0]
        self.parent[b] = a
        self.cnt[a] += self.cnt[b]
        sa, sb = self.sum[a], self.sum[b]
        self.sum[a] = [sa[0] + sb[0], sa[1] + sb[1], sa[2] + sb[2]]
        self.sq[a] += self.sq[b]
        self.perim[a] += self.perim[b] - 2.0 * shared
        self.border[a] += self.border[b]
        self.trans[a] += self.trans[b]
        if self.protected[a] is None:
            self.protected[a] = self.protected[b]
        nba = self.nb[a]
        nba.pop(b, None)
        for j, e in self.nb[b].items():
            if j == a:
                continue
            nbj = self.nb[j]
            nbj.pop(b, None)
            ea = nba.get(j)
            if ea is None:
                nba[j] = e
                nbj[a] = e
            else:
                ea[0] += e[0]
                ea[1] += e[1]
        self.nb[b] = {}
        return a

    def root_labels(self, labels):
        roots = np.array([self.find(i) for i in range(self.n)], dtype=np.int64)
        out = np.where(labels >= 0, roots[np.maximum(labels, 0)], -1)
        return out

    # -- 4. hierarchical merging -------------------------------------------
    def merge_cost(self, a, b):
        cfg = self.cfg
        pa, pb = self.protected[a], self.protected[b]
        if pa is not None and pb is not None and pa != pb:
            return math.inf
        ma, mb = self.mean(a), self.mean(b)
        dE = pde(ma, mb)
        na, nb_ = self.cnt[a], self.cnt[b]
        # uncertainty of small-region means (noise / sqrt(n))
        se = 2.0 * self.noise * math.sqrt(1.0 / na + 1.0 / nb_)
        color = max(0.0, dE - se) / cfg.tolerance
        e = self.nb[a].get(b)
        if e is None:
            return math.inf
        contrast = e[1] / e[0]
        edge = max(0.0, contrast - 2.0 * self.noise - 1.0) / cfg.tolerance
        # region consistency: rise in internal variance when merged
        n = na + nb_
        ra, rb = self.raw_mean(a), self.raw_mean(b)
        d2 = (ra[0] - rb[0]) ** 2 + (ra[1] - rb[1]) ** 2 + (ra[2] - rb[2]) ** 2
        var_inc = math.sqrt(na * nb_ / (n * n) * d2) / cfg.tolerance
        # small regions are cheap to merge (they cannot carry structure)
        w_var = 0.35 * min(1.0, min(na, nb_) / 200.0)
        return color + 0.5 * cfg.edge_preservation * edge + w_var * var_inc

    def _initial_heap(self, threshold):
        """Vectorised merge_cost over every adjacent pair (same formula)."""
        cfg = self.cfg
        I, J, BL, BS = [], [], [], []
        for i in self.roots():
            for j, e in self.nb[i].items():
                if j > i:
                    I.append(i)
                    J.append(j)
                    BL.append(e[0])
                    BS.append(e[1])
        if not I:
            return []
        I = np.array(I)
        J = np.array(J)
        cnt = np.array(self.cnt)
        sm = np.array(self.sum)
        raw = sm / cnt[:, None]
        mean = raw.copy()
        for k, pc in enumerate(self.protected):
            if pc is not None:
                mean[k] = pc
        if self.fixed is not None:
            fx = np.array(self.fixed)
            mean = np.where(np.array([p is None for p in self.protected])[:, None], fx, mean)
        dE = pde_vec(mean[I], mean[J])
        na, nb_ = cnt[I], cnt[J]
        se = 2.0 * self.noise * np.sqrt(1.0 / na + 1.0 / nb_)
        color = np.maximum(0.0, dE - se) / cfg.tolerance
        contrast = np.array(BS) / np.array(BL)
        edge = np.maximum(0.0, contrast - 2.0 * self.noise - 1.0) / cfg.tolerance
        n = na + nb_
        d2 = ((raw[I] - raw[J]) ** 2).sum(1)
        var_inc = np.sqrt(na * nb_ / (n * n) * d2) / cfg.tolerance
        w_var = 0.35 * np.minimum(1.0, np.minimum(na, nb_) / 200.0)
        cost = color + 0.5 * cfg.edge_preservation * edge + w_var * var_inc
        # protected pairs with different locked colours never merge
        prot = [k for k, pc in enumerate(self.protected) if pc is not None]
        if prot:
            pa = np.array([self.protected[k] is not None for k in range(self.n)])
            both = pa[I] & pa[J]
            for idx in np.nonzero(both)[0].tolist():
                if self.protected[I[idx]] != self.protected[J[idx]]:
                    cost[idx] = np.inf
        keep = cost <= threshold
        heap = list(zip(cost[keep].tolist(), I[keep].tolist(), J[keep].tolist(),
                        [0] * int(keep.sum()), [0] * int(keep.sum())))
        heapq.heapify(heap)
        return heap

    def hierarchical_merge(self, threshold=1.0, max_passes=3):
        merges = 0
        for _ in range(max_passes):
            ver = [0] * self.n
            heap = self._initial_heap(threshold)
            pass_merges = 0
            while heap:
                c, i, j, vi, vj = heapq.heappop(heap)
                ri, rj = self.find(i), self.find(j)
                if ri == rj or rj not in self.nb[ri]:
                    continue
                if ri != i or rj != j or ver[ri] != vi or ver[rj] != vj:
                    # stale entry: re-evaluate lazily
                    c2 = self.merge_cost(ri, rj)
                    if c2 <= threshold:
                        heapq.heappush(heap, (c2, ri, rj, ver[ri], ver[rj]))
                    continue
                r = self.merge(ri, rj)
                ver[r] += 1
                pass_merges += 1
                for idx, k in enumerate(self.nb[r]):
                    if idx >= 8:
                        break  # remaining neighbours are refreshed lazily
                    c3 = self.merge_cost(r, k)
                    if c3 <= threshold:
                        heapq.heappush(heap, (c3, r, k, ver[r], ver[k]))
            merges += pass_merges
            if pass_merges == 0:
                break
        return merges

    # -- 5b. ringing / halo strips ------------------------------------------
    def absorb_halos(self, width_max, de_max):
        """Thin strips hugging a similar-coloured region (JPEG ringing, chroma
        bleed) are folded into that region."""
        merged = 0
        cand = [r for r in self.roots() if self.nb[r]]
        cand.sort(key=lambda r: self.cnt[r])
        for r in cand:
            if self.parent[r] != r or self.protected[r] is not None or not self.nb[r]:
                continue
            per = max(self.perim[r], 1.0)
            width = 2.0 * self.cnt[r] / per
            if width > width_max:
                continue
            mr = self.mean(r)
            best, best_de = None, math.inf
            for j, e in self.nb[r].items():
                if e[0] < 0.2 * per or self.cnt[j] < self.cnt[r]:
                    continue
                dE = pde(mr, self.mean(j))
                if dE < best_de:
                    best, best_de = j, dE
            if best is not None and best_de <= de_max:
                self.merge(r, best)
                merged += 1
        return merged

    def _mix_score(self, i, mi, tol):
        """1 if region i's colour is explained as a blend of two neighbours
        (an antialiasing remnant), else 0..1 falloff."""
        nbs = sorted(self.nb[i].items(), key=lambda kv: -kv[1][0])[:3]
        if len(nbs) < 2:
            return 0.0
        c = np.array(self.raw_mean(i))
        best = 0.0
        for x in range(len(nbs)):
            for y in range(x + 1, len(nbs)):
                c1 = np.array(self.mean(nbs[x][0]))
                c2 = np.array(self.mean(nbs[y][0]))
                d = c1 - c2
                if float(d @ d) < (2 * tol) ** 2:
                    continue
                a, resid = blend_fit(c, c1, c2)
                if not 0.0 < a < 1.0:
                    continue
                best = max(best, math.exp(-(resid**2) / (2 * tol * tol)))
        return best

    # -- 6. island absorption ----------------------------------------------
    def absorb_islands(self, max_area, sensitivity, protect_holes=True):
        cfg = self.cfg
        tol = cfg.tolerance
        b0 = -3.0 + 0.6 * sensitivity
        absorbed = 0
        protected_kept = 0
        for _ in range(2):
            cand = [i for i in self.roots() if self.cnt[i] <= max_area and self.nb[i]]
            cand.sort(key=lambda i: self.cnt[i])
            changed = 0
            for i in cand:
                if self.parent[i] != i or self.cnt[i] > max_area or not self.nb[i]:
                    continue
                if self.protected[i] is not None:
                    continue
                mi = self.mean(i)
                per = max(self.perim[i], 1.0)
                best, best_cost, best_de, best_edge = None, math.inf, 0.0, 0.0
                for j, e in self.nb[i].items():
                    dE = pde(mi, self.mean(j))
                    contrast = e[1] / e[0]
                    cost = dE / tol + 0.5 * contrast / tol + 1.0 * (1.0 - e[0] / per)
                    if cost < best_cost:
                        best, best_cost, best_de, best_edge = j, cost, dE, contrast
                if best is None:
                    continue
                area = self.cnt[i]
                s_small = max(0.0, 1.0 - area / (max_area + 1.0))
                s_sim = math.exp(-(best_de**2) / (2.0 * (1.5 * tol) ** 2))
                iso = per * per / (4.0 * math.pi * area)
                s_irr = min(1.0, max(0.0, (iso - 1.3) / 3.0))
                s_edge = min(1.0, max(0.0, best_edge - 2 * self.noise) / (4.0 * tol))
                s_mix = self._mix_score(i, mi, tol)
                z = (b0 + 2.0 * s_small + 4.0 * s_sim + 1.0 * s_irr + 3.0 * s_mix
                     - 3.0 * cfg.edge_preservation * s_edge)
                p = 1.0 / (1.0 + math.exp(-z))
                enclosed = len(self.nb[i]) == 1 and self.border[i] == 0
                if (
                    protect_holes
                    and enclosed
                    and best_de > 2.0 * tol
                    and iso < 2.5
                    and area >= 4
                ):
                    protected_kept += 1
                    continue  # compact, distinct enclosed shape: eye, dot, counter
                if best_de > 2.0 * tol and area > 3 + sensitivity and s_mix < 0.5:
                    protected_kept += 1
                    continue  # high-contrast shape (glyph, dot): never an artifact
                if p > 0.5 or area <= 1 + sensitivity // 4:
                    self.merge(i, best)
                    changed += 1
            absorbed += changed
            if not changed:
                break
        return absorbed, protected_kept


def blend_fit(c, c1, c2, chroma_weight=1.0):
    """
    Is Lab colour c an antialiasing blend of c1 and c2? Renderers blend in
    gamma-encoded sRGB, so project there; report the residual in Lab units.
    Returns (alpha, residual) with c ~ alpha*c1 + (1-alpha)*c2.
    """
    rgb = lab_to_srgb(np.array([c, c1, c2], float))
    x, x1, x2 = rgb
    d = x1 - x2
    dd = float(d @ d)
    if dd < 1e-9:
        return 0.0, math.inf
    a = float((x - x2) @ d) / dd
    a_c = min(1.0, max(0.0, a))
    blend = srgb_to_lab(x2 + a_c * d)
    diff = np.asarray(c) - blend
    diff[1:] *= chroma_weight
    return a, float(np.linalg.norm(diff))


def _wdist(c1, c2, chroma_weight):
    d = np.asarray(c1, float) - np.asarray(c2, float)
    d[1:] *= chroma_weight
    return float(np.linalg.norm(d))


# --------------------------------------------------------------------------
# 5. Antialiasing transition strips
# --------------------------------------------------------------------------


def dissolve_transitions(labels, g, lab, cfg):
    """Split thin regions whose colour is a mix of two neighbours."""
    thin_max = 2.0 + 2.0 * (1.0 - cfg.edge_preservation)
    tol = cfg.tolerance
    n = g.n
    is_t = np.zeros(n, bool)
    N1 = np.zeros(n, np.int64)
    N2 = np.zeros(n, np.int64)
    C1 = np.zeros((n, 3))
    C2 = np.zeros((n, 3))
    count = 0
    for r in g.roots():
        if g.protected[r] is not None or not g.nb[r]:
            continue
        width = 2.0 * g.cnt[r] / max(g.perim[r], 1.0)
        if width > thin_max:
            continue
        if g.trans[r] > 0:
            # matte fringe: blend of one neighbour with the white/black matte
            # the image was flattened on before its alpha was cut out
            j1, e1 = max(g.nb[r].items(), key=lambda kv: kv[1][0])
            if e1[0] + g.trans[r] >= 0.6 * g.perim[r]:
                c = np.array(g.raw_mean(r))
                c1 = np.array(g.mean(j1))
                for matte in (np.array([100.0, 0, 0]), np.array([0.0, 0, 0])):
                    d = c1 - matte
                    if float(d @ d) < (2.0 * tol) ** 2:
                        continue
                    alpha, resid = blend_fit(c, c1, matte)
                    if 0.05 < alpha < 0.98 and resid <= 0.9 * tol + 2 * g.noise:
                        is_t[r], N1[r], N2[r], C1[r], C2[r] = True, j1, j1, c1, c1
                        count += 1
                        break
                if is_t[r]:
                    continue
        if len(g.nb[r]) < 2:
            continue
        nbs = sorted(g.nb[r].items(), key=lambda kv: -kv[1][0])[:2]
        (j1, e1), (j2, e2) = nbs
        if e1[0] + e2[0] < 0.6 * g.perim[r]:
            continue
        c = np.array(g.raw_mean(r))
        c1 = np.array(g.mean(j1))
        c2 = np.array(g.mean(j2))
        d = c1 - c2
        if float(d @ d) < (2.0 * tol) ** 2:
            continue
        alpha, resid = blend_fit(c, c1, c2)
        if not 0.05 < alpha < 0.95:
            continue
        if resid > 0.9 * tol + 2 * g.noise:
            continue
        is_t[r], N1[r], N2[r], C1[r], C2[r] = True, j1, j2, c1, c2
        count += 1
    if not count:
        return labels, 0
    roots = g.root_labels(labels)
    m = (roots >= 0) & is_t[np.maximum(roots, 0)]
    rl = roots[m]
    px = lab[m]
    d1 = ((px - C1[rl]) ** 2).sum(1)
    d2 = ((px - C2[rl]) ** 2).sum(1)
    roots = roots.copy()
    roots[m] = np.where(d1 <= d2, N1[rl], N2[rl])
    return roots, count


# --------------------------------------------------------------------------
# 7. Palette
# --------------------------------------------------------------------------


def label_medians(labels, lab, n):
    """Per-label, per-channel median of Lab values."""
    m = labels.ravel() >= 0
    lbl = labels.ravel()[m]
    vals = lab.reshape(-1, 3)[m]
    cnt = np.bincount(lbl, minlength=n)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    mid = start + np.maximum(cnt - 1, 0) // 2
    out = np.zeros((n, 3))
    for c in range(3):
        order = np.lexsort((vals[:, c], lbl))
        sv = vals[order, c]
        ok = cnt > 0
        out[ok, c] = sv[mid[ok]]
    return out, cnt


def _is_blend_of(c, pal, tol):
    """True if c is (approximately) an sRGB blend of two palette colours."""
    if len(pal) < 2:
        return False
    for x in range(len(pal)):
        for y in range(x + 1, len(pal)):
            if float(((pal[x] - pal[y]) ** 2).sum()) < (3 * tol) ** 2:
                continue
            a, res = blend_fit(c, pal[x], pal[y])
            if 0.12 < a < 0.88 and res <= 0.6 * tol:
                return True
    return False


def build_palette(colors, weights, cfg, protected_lab):
    """
    Region-level palette. colors (R,3) Lab, weights (R,). Returns
    (palette (K,3), assignment (R,), locked (K,) bool).
    """
    tol = cfg.tolerance
    R = len(colors)
    order = np.argsort(-weights)
    pal = [np.asarray(p, float) for p in protected_lab]
    locked = [True] * len(pal)
    if cfg.target_colors and cfg.target_colors > 0:
        k = max(cfg.target_colors, len(pal))
        if not pal:
            pal.append(colors[order[0]].copy())
            locked.append(False)
        P = np.array(pal)
        while len(pal) < min(k, R):
            d = pde_vec(colors[:, None, :], P[None, :, :]).min(1)
            score = weights * d**2
            j = int(np.argmax(score))
            if d[j] < 0.5:
                break
            pal.append(colors[j].copy())
            locked.append(False)
            P = np.array(pal)
    else:
        total = float(weights.sum())
        min_support = max(12.0, 2e-4 * total)
        for i in order:
            c = colors[i]
            if weights[i] < min_support:
                break  # order is by weight: the rest are too small to found a colour
            if pal and pde_vec(c[None, :], np.array(pal)).min() <= tol:
                continue
            if weights[i] < 0.005 * total and _is_blend_of(c, pal, tol):
                continue  # antialias / bleed colour between two palette entries
            pal.append(c.copy())
            locked.append(False)
        if not pal:
            pal.append(colors[order[0]].copy())
            locked.append(False)
    P = np.array(pal)
    locked = np.array(locked)
    # weighted Lloyd refinement (Euclidean Lab for the update step)
    for _ in range(8):
        assign = np.argmin(((colors[:, None, :] - P[None]) ** 2).sum(-1), 1)
        newP = P.copy()
        for k in range(len(P)):
            if locked[k]:
                continue
            m = assign == k
            if m.any():
                w = weights[m]
                newP[k] = (colors[m] * w[:, None]).sum(0) / w.sum()
        if np.allclose(newP, P, atol=1e-3):
            break
        P = newP
    dmat = pde_vec(colors[:, None, :], P[None])
    assign = np.argmin(dmat, 1)
    # "target is desired, not mandatory": keep distinct colours that would be lost
    dmin = dmat[np.arange(R), assign]
    min_keep = max(16.0, 0.0005 * weights.sum())
    extra = [i for i in order if dmin[i] > 2.5 * tol and weights[i] >= min_keep]
    for i in extra:
        c = colors[i]
        if pde_vec(c[None], P).min() > 2.5 * tol and not _is_blend_of(c, list(P), tol):
            P = np.vstack([P, c])
            locked = np.append(locked, False)
    # merge redundant entries
    keep = list(range(len(P)))
    merged = True
    while merged:
        merged = False
        for x in range(len(keep)):
            for y in range(x + 1, len(keep)):
                a, b = keep[x], keep[y]
                if locked[a] and locked[b]:
                    continue
                if pde(tuple(P[a]), tuple(P[b])) < 0.5 * tol:
                    if locked[b] and not locked[a]:
                        a, b = b, a
                    keep.remove(b)
                    merged = True
                    break
            if merged:
                break
    P = P[keep]
    locked = locked[keep]
    dmat = pde_vec(colors[:, None, :], P[None])
    assign = np.argmin(dmat, 1)
    used = np.unique(assign)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    return P[used], remap[assign], locked[used]


# --------------------------------------------------------------------------
# 8. Tracing: planar boundary graph -> shared chains -> Bezier
# --------------------------------------------------------------------------

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # right, down, left, up (y down)


def trace_regions(labels):
    """
    Returns (chains, rings) where chains[k] = (pts, left, right, closed)
    and rings[label] = list of rings, each ring = list of (chain, reversed).
    Each region boundary is oriented with the region on its left.
    """
    H, W = labels.shape
    P = np.full((H + 2, W + 2), -1, np.int64)
    P[1:-1, 1:-1] = labels
    Hz = P[0 : H + 1, 1 : W + 1] != P[1 : H + 2, 1 : W + 1]  # (H+1, W)
    Vt = P[1 : H + 1, 0 : W + 1] != P[1 : H + 1, 1 : W + 2]  # (H, W+1)
    deg = np.zeros((H + 1, W + 1), np.int8)
    deg[:, :-1] += Hz
    deg[:, 1:] += Hz
    deg[:-1, :] += Vt
    deg[1:, :] += Vt
    Hl = Hz.tolist()
    Vl = Vt.tolist()
    Pl = P.tolist()
    J = (deg >= 3).tolist()
    nH = (H + 1) * W

    def edge(r, c, d):
        """edge id and target vertex for leaving (r,c) in direction d, or None."""
        if d == 0:
            if c < W and Hl[r][c]:
                return r * W + c, r, c + 1
        elif d == 2:
            if c > 0 and Hl[r][c - 1]:
                return r * W + c - 1, r, c - 1
        elif d == 1:
            if r < H and Vl[r][c]:
                return nH + r * (W + 1) + c, r + 1, c
        else:
            if r > 0 and Vl[r - 1][c]:
                return nH + (r - 1) * (W + 1) + c, r - 1, c
        return None

    def sides(r, c, d):
        if d == 0:
            return Pl[r][c + 1], Pl[r + 1][c + 1]
        if d == 2:
            return Pl[r + 1][c], Pl[r][c]
        if d == 1:
            return Pl[r + 1][c + 1], Pl[r + 1][c]
        return Pl[r][c], Pl[r][c + 1]

    visited = bytearray(nH + H * (W + 1))
    chains = []

    def walk(r0, c0, d):
        left, right = sides(r0, c0, d)
        pts = [(c0, r0)]
        dirs = [d]
        r, c = r0, c0
        while True:
            eid, r, c = edge(r, c, d)
            visited[eid] = 1
            pts.append((c, r))
            if J[r][c] or (r == r0 and c == c0):
                break
            back = (d + 2) & 3
            for nd in (d, (d + 1) & 3, (d + 3) & 3):
                if nd != back and edge(r, c, nd) is not None:
                    d = nd
                    break
            dirs.append(d)
        closed = pts[0] == pts[-1] and not J[r0][c0]
        chains.append((pts, left, right, closed, dirs[0], dirs[-1]))

    jr, jc = np.nonzero(deg >= 3)
    for r, c in zip(jr.tolist(), jc.tolist()):
        for d in range(4):
            e = edge(r, c, d)
            if e is not None and not visited[e[0]]:
                walk(r, c, d)
    # junction-free loops
    for eid in np.nonzero(np.concatenate([Hz.ravel(), Vt.ravel()]))[0].tolist():
        if visited[eid]:
            continue
        if eid < nH:
            r, c = divmod(eid, W)
            walk(r, c, 0)
        else:
            r, c = divmod(eid - nH, W + 1)
            walk(r, c, 1)

    # assemble rings per region
    per_region = {}
    for k, (pts, left, right, closed, d0, d1) in enumerate(chains):
        for lab_, rev in ((left, False), (right, True)):
            if lab_ < 0:
                continue
            if rev:
                start, end, sd, ed = pts[-1], pts[0], (d1 + 2) & 3, (d0 + 2) & 3
            else:
                start, end, sd, ed = pts[0], pts[-1], d0, d1
            per_region.setdefault(lab_, []).append((k, rev, start, end, sd, ed, closed))

    rings = {}
    for lab_, items in per_region.items():
        out = []
        by_start = {}
        for it in items:
            if it[6]:
                out.append([(it[0], it[1])])
            else:
                by_start.setdefault(it[2], []).append(it)
        used = set()
        for it in items:
            if it[6] or (it[0], it[1]) in used:
                continue
            first = it
            ring = [(it[0], it[1])]
            used.add((it[0], it[1]))
            v, d = it[3], it[5]
            guard = 0
            while guard < 1_000_000:
                guard += 1
                cands = [x for x in by_start.get(v, []) if (x[0], x[1]) not in used]
                if v == first[2]:
                    cands.append(first)
                if not cands:
                    break
                pick = None
                for want in ((d + 3) & 3, d, (d + 1) & 3, (d + 2) & 3):
                    for x in cands:
                        if x[4] == want:
                            pick = x
                            break
                    if pick:
                        break
                if pick is None or pick is first:
                    break
                ring.append((pick[0], pick[1]))
                used.add((pick[0], pick[1]))
                v, d = pick[3], pick[5]
            out.append(ring)
        rings[lab_] = out
    return chains, rings


def _dp(points, tol):
    """Douglas-Peucker keeping endpoints; returns indices."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    t2 = tol * tol
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ax, ay = points[a]
        bx, by = points[b]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            px, py = points[i]
            if L2 == 0:
                d2 = (px - ax) ** 2 + (py - ay) ** 2
            else:
                cr = dx * (py - ay) - dy * (px - ax)
                d2 = cr * cr / L2
            if d2 > best:
                best, bi = d2, i
        if best > t2:
            keep[bi] = True
            stack.append((a, bi))
            stack.append((bi, b))
    return [i for i in range(n) if keep[i]]


def _turn(p0, p1, p2):
    a1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    t = a2 - a1
    while t > math.pi:
        t -= 2 * math.pi
    while t < -math.pi:
        t += 2 * math.pi
    return t


def _unit(x, y):
    L = math.hypot(x, y)
    return (x / L, y / L) if L > 1e-12 else (0.0, 0.0)


def _line_intersect(p1, p2, p3, p4):
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _find_corners(p, closed, cfg):
    """
    Corner detection at a scale of k samples (single-pixel jaggies are not
    corners). Each run of high-turn samples becomes one corner, snapped to
    the intersection of the incoming and outgoing edges when that is close,
    which turns pixel chamfers back into crisp vertices.
    Returns (points, anchor_indices).
    """
    n = len(p)
    k = max(1, min(3, n // 5))
    thr = math.radians(cfg.corner_angle)
    hot = [False] * n
    for i in range(n):
        if closed:
            a, b = p[(i - k) % n], p[(i + k) % n]
        else:
            if i < k or i >= n - k:
                continue
            a, b = p[i - k], p[i + k]
        if abs(_turn(a, p[i], b)) > thr:
            hot[i] = True
    # group consecutive hot samples into runs
    runs = []
    i = 0
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs[0][0] = runs[-1][0] - n
        runs.pop()
    if closed and runs and len(runs) == 1 and runs[0][1] - runs[0][0] >= n - 1:
        return list(p), []  # everything "hot": tiny loop, treat as smooth
    out = []
    anchors = []
    replace = {}
    drop = set()
    for s0, e0 in runs:
        g = lambda t: p[t % n] if closed else p[max(0, min(n - 1, t))]
        X = _line_intersect(g(s0 - k), g(s0), g(e0), g(e0 + k))
        mid = (s0 + e0) // 2
        if X is not None:
            cx = sum(g(t)[0] for t in range(s0, e0 + 1)) / (e0 - s0 + 1)
            cy = sum(g(t)[1] for t in range(s0, e0 + 1)) / (e0 - s0 + 1)
            if math.hypot(X[0] - cx, X[1] - cy) > 2.0:
                X = None
        replace[mid % n] = X if X is not None else g(mid)
        for t in range(s0, e0 + 1):
            if t % n != mid % n:
                drop.add(t % n)
    for i in range(n):
        if i in drop:
            continue
        if i in replace:
            anchors.append(len(out))
            out.append(replace[i])
        else:
            out.append(p[i])
    if not closed:
        if not anchors or anchors[0] != 0:
            anchors.insert(0, 0)
        if anchors[-1] != len(out) - 1:
            anchors.append(len(out) - 1)
    return out, anchors


def _smooth(p, anchors, closed, iterations):
    n = len(p)
    pin = set(anchors)
    q = list(p)
    for _ in range(iterations):
        r = list(q)
        for i in range(n):
            if i in pin or (not closed and (i == 0 or i == n - 1)):
                continue
            a, b = q[(i - 1) % n], q[(i + 1) % n]
            r[i] = ((a[0] + 2 * q[i][0] + b[0]) * 0.25, (a[1] + 2 * q[i][1] + b[1]) * 0.25)
        q = r
    return q


# ---- Schneider least-squares cubic fitting -------------------------------


def _bez_eval(b, u):
    mu = 1.0 - u
    b0 = mu * mu * mu
    b1 = 3 * u * mu * mu
    b2 = 3 * u * u * mu
    b3 = u * u * u
    return b0[:, None] * b[0] + b1[:, None] * b[1] + b2[:, None] * b[2] + b3[:, None] * b[3]


def _gen_bezier(d, u, t1, t2):
    p0, p3 = d[0], d[-1]
    mu = 1.0 - u
    B0, B1, B2, B3 = mu**3, 3 * u * mu**2, 3 * u**2 * mu, u**3
    A1 = B1[:, None] * t1
    A2 = B2[:, None] * t2
    C00 = (A1 * A1).sum()
    C01 = (A1 * A2).sum()
    C11 = (A2 * A2).sum()
    tmp = d - ((B0 + B1)[:, None] * p0 + (B2 + B3)[:, None] * p3)
    X0 = (A1 * tmp).sum()
    X1 = (A2 * tmp).sum()
    det = C00 * C11 - C01 * C01
    seg = float(np.hypot(*(p3 - p0)))
    eps = 1e-6 * max(seg, 1e-9)
    if abs(det) > 1e-12:
        al = (X0 * C11 - X1 * C01) / det
        ar = (C00 * X1 - C01 * X0) / det
    else:
        al = ar = -1.0
    if al < eps or ar < eps or al > 2 * seg + 2 or ar > 2 * seg + 2:
        al = ar = seg / 3.0
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _reparam(d, u, b):
    q = _bez_eval(b, u)
    q1 = np.array([3 * (b[1] - b[0]), 3 * (b[2] - b[1]), 3 * (b[3] - b[2])])
    q2 = np.array([6 * (q1[1] - q1[0]) / 3, 6 * (q1[2] - q1[1]) / 3]) / 2
    mu = 1.0 - u
    d1 = (mu * mu)[:, None] * q1[0] + (2 * u * mu)[:, None] * q1[1] + (u * u)[:, None] * q1[2]
    d2 = mu[:, None] * q2[0] * 2 + u[:, None] * q2[1] * 2
    diff = q - d
    num = (diff * d1).sum(1)
    den = (d1 * d1).sum(1) + (diff * d2).sum(1)
    nu = np.where(np.abs(den) > 1e-12, u - num / np.where(den == 0, 1, den), u)
    nu = np.clip(nu, 0.0, 1.0)
    nu[0], nu[-1] = 0.0, 1.0
    return nu


def _fit_cubic(d, t1, t2, err, depth=0):
    n = len(d)
    if n == 2:
        dist = float(np.hypot(*(d[1] - d[0]))) / 3.0
        return [np.array([d[0], d[0] + t1 * dist, d[1] + t2 * dist, d[1]])]
    seglen = np.hypot(*(np.diff(d, axis=0).T))
    u = np.concatenate([[0.0], np.cumsum(seglen)])
    u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, n)
    b = _gen_bezier(d, u, t1, t2)
    e2 = ((_bez_eval(b, u) - d) ** 2).sum(1)
    split = int(np.argmax(e2))
    if e2[split] < err * err:
        return [b]
    if e2[split] < (4 * err) ** 2:
        for _ in range(6):
            u = _reparam(d, u, b)
            b = _gen_bezier(d, u, t1, t2)
            e2 = ((_bez_eval(b, u) - d) ** 2).sum(1)
            split = int(np.argmax(e2))
            if e2[split] < err * err:
                return [b]
    if depth > 24:
        return [b]
    split = min(max(split, 1), n - 2)
    tc = d[split - 1] - d[split + 1]
    L = float(np.hypot(*tc))
    tc = tc / L if L > 1e-12 else t2
    return _fit_cubic(d[: split + 1], t1, tc, err, depth + 1) + _fit_cubic(
        d[split:], -tc, t2, err, depth + 1
    )


def _end_tangent(d, forward=True, look=3):
    m = min(look, len(d) - 1)
    v = d[m] - d[0] if forward else d[-1 - m] - d[-1]
    L = float(np.hypot(*v))
    return v / L if L > 1e-12 else np.array([1.0, 0.0])


def _fit_span(Q, err, t1=None, t2=None):
    d = np.asarray(Q, float)
    if len(d) == 2:
        return [(tuple(d[0]), None, None, tuple(d[1]))]
    # straight?
    p0, p1 = d[0], d[-1]
    v = p1 - p0
    L = float(np.hypot(*v))
    if L > 1e-9:
        dist = np.abs(v[0] * (d[:, 1] - p0[1]) - v[1] * (d[:, 0] - p0[0])) / L
        if dist.max() <= 0.9 * err:
            return [(tuple(p0), None, None, tuple(p1))]
    t1 = _end_tangent(d, True) if t1 is None else t1
    t2 = _end_tangent(d, False) if t2 is None else t2
    out = []
    for b in _fit_cubic(d, t1, t2, err):
        out.append((tuple(b[0]), tuple(b[1]), tuple(b[2]), tuple(b[3])))
    return out


def fit_chain(pts, closed, straight, cfg):
    """Return list of segments (p0, c1, c2, p3); c1 is None for straight lines."""
    if straight or len(pts) == 2:
        idx = _dp(pts, 0.01)
        q = [pts[i] for i in idx]
        return [(q[i], None, None, q[i + 1]) for i in range(len(q) - 1)]
    n = len(pts)
    if closed:
        base = pts[:-1]
        m = len(base)
        mids = [((base[i][0] + base[(i + 1) % m][0]) / 2.0,
                 (base[i][1] + base[(i + 1) % m][1]) / 2.0) for i in range(m)]
    else:
        mids = [pts[0]] + [((pts[i][0] + pts[i + 1][0]) / 2.0, (pts[i][1] + pts[i + 1][1]) / 2.0)
                           for i in range(n - 1)] + [pts[-1]]
    q, anchors = _find_corners(mids, closed, cfg)
    q = _smooth(q, anchors, closed, max(0, int(round(4 * cfg.smoothness))))
    err = cfg.trace_tolerance
    segs = []
    if closed:
        m = len(q)
        if m < 3:
            return []
        if not anchors:
            # smooth closed loop: split in two halves with matching tangents
            h = m // 2
            ring = q + [q[0]]
            ta = _unit(q[1][0] - q[-1][0], q[1][1] - q[-1][1])
            tb = _unit(q[h + 1][0] - q[h - 1][0], q[h + 1][1] - q[h - 1][1])
            ta, tb = np.array(ta), np.array(tb)
            segs += _fit_span(ring[: h + 1], err, ta, -tb)
            segs += _fit_span(ring[h:], err, tb, -ta)
            return segs
        rot = anchors[0]
        q = q[rot:] + q[:rot]
        anchors = [(a - rot) % m for a in anchors] + [m]
        q = q + [q[0]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b > a:
            segs += _fit_span(q[a : b + 1], err)
    return segs


def _fmt(v):
    s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def rings_to_path(ring_list, chain_segs):
    parts = []
    for ring in ring_list:
        segs = []
        for k, rev in ring:
            s = chain_segs[k]
            if rev:
                s = [(p3, c2, c1, p0) for (p0, c1, c2, p3) in reversed(s)]
            segs.extend(s)
        if not segs:
            continue
        p = segs[0][0]
        out = ["M%s,%s" % (_fmt(p[0]), _fmt(p[1]))]
        last = len(segs) - 1
        for i, (p0, c1, c2, p3) in enumerate(segs):
            if c1 is None:
                if i != last:  # closing line is implied by Z
                    out.append("L%s,%s" % (_fmt(p3[0]), _fmt(p3[1])))
            else:
                out.append("C%s,%s %s,%s %s,%s" % tuple(_fmt(v) for v in (*c1, *c2, *p3)))
        out.append("Z")
        parts.append("".join(out))
    return " ".join(parts)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


class Result:
    pass


def deartifact(rgba, cfg, log=None):
    """
    rgba: uint8 (H,W,4). Returns Result with:
        labels      (H,W) final region labels (-1 transparent)
        region_class  (R,) palette index per region
        palette_lab (K,3), palette_hex [K], locked [K]
        cleaned_rgba  (H,W,4) uint8 rendered result
        info        dict of statistics
    """
    log = log or (lambda *a: None)
    rgb8 = rgba[..., :3]
    valid = rgba[..., 3] >= cfg.alpha_threshold
    lab = srgb_to_lab(rgb8 / 255.0).astype(np.float32)
    info = analyse(rgb8, lab, valid)
    log("analysis", info)

    # 2. restoration
    restored, noise_mask = restore(lab, info, cfg)
    info["noise_pixels"] = int(noise_mask.sum())

    # raw crack contrast (unsmoothed image) used as boundary evidence
    ch = np.sqrt(((lab[:, 1:] - lab[:, :-1]) ** 2).sum(-1))
    cv = np.sqrt(((lab[1:, :] - lab[:-1, :]) ** 2).sum(-1))

    protected_lab = []
    for h in cfg.protected_colors:
        try:
            protected_lab.append(tuple(hex_to_lab(h).tolist()))
        except ValueError:
            pass

    # 3. over-segmentation
    labels, step = initial_segmentation(restored, valid, cfg, info)
    info["superpixel_step"] = step
    info["initial_regions"] = int(labels.max() + 1)

    # 4. region graph merging
    g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg)
    if protected_lab:
        _mark_protected(g, protected_lab, cfg.tolerance)
    info["merges"] = g.hierarchical_merge()
    labels = relabel_connected(g.root_labels(labels))

    # 5. antialiasing strips
    info["transition_strips"] = 0
    for _ in range(3):  # 2-3 px transitions are peeled one ring at a time
        g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg)
        labels, k = dissolve_transitions(labels, g, restored, cfg)
        labels = relabel_connected(labels)
        info["transition_strips"] += k
        if not k:
            break
    g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg)
    if protected_lab:
        _mark_protected(g, protected_lab, cfg.tolerance)
    info["halo_strips"] = g.absorb_halos(
        2.0 + 2.0 * (1.0 - cfg.edge_preservation),
        cfg.tolerance * (1.0 + 1.5 * (1.0 - cfg.edge_preservation)),
    )
    labels = relabel_connected(g.root_labels(labels))

    # 6. islands
    HW = int(valid.sum())
    max_area = cfg.island_max_area or int(
        max(4, min(200, (cfg.island_sensitivity + 1) ** 2 * (1 + HW / 2_000_000.0)))
    )
    info["island_max_area"] = max_area
    g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg)
    if protected_lab:
        _mark_protected(g, protected_lab, cfg.tolerance)
    g.hierarchical_merge()
    info["islands_absorbed"], info["islands_protected"] = g.absorb_islands(
        max_area, cfg.island_sensitivity, cfg.protect_holes
    )
    labels = relabel_connected(g.root_labels(labels))

    # 7. palette (region level)
    n = int(labels.max() + 1)
    med, cnt = label_medians(labels, restored, n)
    g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg)
    width = 2.0 * cnt / np.maximum(np.array(g.perim), 1.0)
    thin = (width < 3.6) & (cnt > 0)
    # thin structures carry blended (partial-volume) colours: keep them out of
    # palette estimation, then unmix them against their background
    weights = cnt.astype(float) * np.where(thin, 0.02, 1.0)
    pal, assign, locked = build_palette(med, weights, cfg, protected_lab)
    pal, assign, locked, ink, info["thin_unmixed"] = assign_thin_regions(
        g, med, thin, pal, assign, locked, cfg, labels, restored
    )
    cls = np.where(labels >= 0, assign[np.maximum(labels, 0)], -1)

    # canonical colour = weighted median of member pixels
    K = len(pal)
    cmed, ccnt = label_medians(cls, restored, K)
    for k in range(K):
        if locked[k] or ink[k]:
            cmed[k] = pal[k]
        elif ccnt[k] == 0:
            cmed[k] = pal[k]

    # boundary refinement (MRF / ICM) against the canonical colours
    lam = 0.4 * cfg.tolerance**2 * (1.25 - 0.5 * cfg.edge_preservation)
    cls = refine_boundaries(cls, restored, cmed, lam)

    # final topology cleanup on the palette map
    labels = relabel_connected(cls)
    n = int(labels.max() + 1)
    region_cls = np.zeros(n, np.int64)
    m = labels >= 0
    region_cls[labels[m]] = cls[m]
    fixed = [tuple(cmed[c]) for c in region_cls.tolist()]
    g = RegionGraph(labels, restored, ch, cv, info["noise"], cfg, fixed_colors=fixed)
    for r in range(n):
        if locked[region_cls[r]]:
            g.protected[r] = tuple(cmed[region_cls[r]])
    final_area = max(2, max_area // 3)
    a2, p2 = g.absorb_islands(final_area, cfg.island_sensitivity, cfg.protect_holes)
    info["islands_absorbed"] += a2
    roots = np.array([g.find(i) for i in range(n)])
    region_cls = region_cls[roots]  # class of the surviving root
    labels = np.where(labels >= 0, roots[np.maximum(labels, 0)], -1)
    cls = np.where(labels >= 0, region_cls[np.maximum(labels, 0)], -1)
    cls, cmed, locked = consolidate_palette(cls, restored, cmed, locked, cfg, keep_color=ink)
    K = len(cmed)
    labels = relabel_connected(cls)
    n = int(labels.max() + 1)
    region_cls = np.zeros(n, np.int64)
    m = labels >= 0
    region_cls[labels[m]] = cls[m]

    res = Result()
    res.labels = labels
    res.region_class = region_cls
    res.palette_lab = cmed
    res.palette_hex = [lab_to_hex(c) for c in cmed]
    res.locked = locked
    res.class_pixels = np.bincount(cls[m], minlength=K)
    rgb = (lab_to_srgb(cmed) * 255).round().astype(np.uint8)
    out = np.zeros_like(rgba)
    out[m, :3] = rgb[cls[m]]
    out[m, 3] = 255
    res.cleaned_rgba = out
    info["final_regions"] = n
    info["palette_size"] = K
    res.info = info
    return res


def assign_thin_regions(g, med, thin, pal, assign, locked, cfg, labels, lab):
    """
    Thin regions (strokes, small text, hairlines) are dominated by
    antialiasing blends with their background, so their mean colour is not
    their ink colour. For each stroke:
      * estimate its ink from its core pixels (farthest from the background);
      * find the palette colour X with ink ~ blend(X, background), weighting
        lightness over chroma (chroma is subsampled in most JPEGs);
      * inks nothing explains are clustered (most reliable observations
        first) into new palette colours, then matched again.
    """
    tol = cfg.tolerance
    CW = 0.25
    pal = [np.asarray(p, float) for p in pal]
    locked = list(locked)
    n_base = len(pal)
    assign = assign.copy()

    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat[flat >= 0], minlength=len(med))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) + int((flat < 0).sum())
    lf = lab.reshape(-1, 3)

    strokes = []  # (r, bgk, ink)
    for r in np.nonzero(thin)[0].tolist():
        if not g.nb[r]:
            continue
        nbs = sorted(g.nb[r].items(), key=lambda kv: -kv[1][0])
        bgk = int(assign[nbs[0][0]])
        bg = pal[bgk]
        if pde(tuple(med[r]), tuple(bg)) <= 0.75 * tol:
            assign[r] = bgk  # indistinguishable from its background
            continue
        px = lf[order[starts[r] : starts[r] + counts[r]]].astype(float)
        d = ((px - bg) ** 2).sum(1)
        ink = px[d >= np.percentile(d, 70)].mean(0)
        strokes.append((r, bgk, ink))

    def match(ink, bgk, palette):
        bg = palette[bgk]
        best_k, best_res = None, math.inf
        for k, X in enumerate(palette):
            if k == bgk or float(((X - bg) ** 2).sum()) < (2 * tol) ** 2:
                continue
            a, res = blend_fit(ink, X, bg, CW)
            if 0.6 <= a <= 1.35 and res < best_res:
                best_k, best_res = k, res
        return best_k, best_res

    unmixed = 0
    unresolved = []
    for r, bgk, ink in strokes:
        k, res = match(ink, bgk, pal)
        if k is not None and res <= 1.2 * tol:
            unmixed += k != assign[r]
            assign[r] = k
        else:
            unresolved.append((r, bgk, ink))

    if unresolved:
        wts = []
        for r, bgk, ink in unresolved:
            bgc = pal[bgk]
            dl = abs(ink[0] - bgc[0]) / 100.0
            chroma_bg = math.hypot(bgc[1], bgc[2])
            wts.append(float(counts[r]) * (0.05 + dl) / (1.0 + chroma_bg / 20.0))
        idx = np.argsort(-np.array(wts))
        centers, csum, cw = [], [], []
        for i in idx.tolist():
            c = unresolved[i][2]
            j = None
            for q, ctr in enumerate(centers):
                if _wdist(ctr, c, CW) <= 1.5 * tol:
                    j = q
                    break
            if j is None:
                centers.append(c.copy())
                csum.append(c * wts[i])
                cw.append(wts[i])
            else:
                csum[j] = csum[j] + c * wts[i]
                cw[j] += wts[i]
                centers[j] = csum[j] / cw[j]
        for ctr in centers:
            if min(_wdist(ctr, p, CW) for p in pal) <= tol:
                continue
            if any(
                0.6 <= a <= 1.35 and res <= 1.2 * tol
                for X in pal for bg in pal if X is not bg
                for a, res in [blend_fit(ctr, X, bg, CW)]
            ):
                continue
            pal.append(ctr)
            locked.append(False)
        for r, bgk, ink in unresolved:
            k, res = match(ink, bgk, pal)
            if k is None:
                d = [_wdist(ink, p, CW) for p in pal]
                d[bgk] = math.inf
                k = int(np.argmin(d))
            assign[r] = k
            unmixed += 1
    ink_mask = np.zeros(len(pal), bool)
    ink_mask[n_base:] = True  # colours estimated from stroke cores: keep exact
    return np.array(pal), assign, np.array(locked), ink_mask, unmixed


def refine_boundaries(cls, lab, pal, lam, iterations=3):
    """
    Boundary refinement: ICM on a Potts MRF restricted to boundary pixels.
    Each pixel picks, among its own class and its 4-neighbours' classes, the
    one minimising |Lab - colour|^2 + lam * (#4-neighbours that disagree).
    (4-neighbourhood: convex corners tie, so colour evidence keeps them sharp.)
    Removes JPEG jaggies on straight/curved edges; thin lines keep their
    pixels because their colour evidence outweighs the smoothness term.
    """
    H, W = cls.shape
    pal = np.asarray(pal, np.float32)
    valid = cls >= 0
    cur = cls.copy()
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(iterations):
        p = np.pad(cur, 1, mode="edge")
        nbr = [p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W] for dy, dx in shifts]
        four = [nbr[1], nbr[6], nbr[3], nbr[4]]
        cands = [cur] + four
        boundary = np.zeros((H, W), bool)
        for c in cands[1:]:
            boundary |= c != cur
        boundary &= valid
        if not boundary.any():
            break
        ys, xs = np.nonzero(boundary)
        px = lab[ys, xs]
        best_cost = None
        best = cur[ys, xs]
        for c in cands:
            cv = c[ys, xs]
            ok = cv >= 0
            cvs = np.maximum(cv, 0)
            data = ((px - pal[cvs]) ** 2).sum(1)
            dis = np.zeros(len(ys), np.float32)
            for nb in four:
                dis += nb[ys, xs] != cv
            cost = np.where(ok, data + lam * dis, np.inf)
            if best_cost is None:
                best_cost = cost
            else:
                upd = cost < best_cost
                best_cost = np.where(upd, cost, best_cost)
                best = np.where(upd, cv, best)
        new = cur.copy()
        new[ys, xs] = best
        if np.array_equal(new, cur):
            break
        cur = new
    return cur


def consolidate_palette(cls, lab, cmed, locked, cfg, keep_color=None):
    """Drop empty classes and fuse classes whose canonical colours ended up
    closer than the tolerance (auto mode) / half of it (fixed count)."""
    K = len(cmed)
    m = cls >= 0
    pix = np.bincount(cls[m], minlength=K)
    limit = cfg.tolerance * (0.5 if cfg.target_colors else 0.8)
    target = np.arange(K)
    order = np.argsort(-pix)
    alive = [k for k in order.tolist() if pix[k] > 0]
    kept = []
    for k in alive:
        best, bd = None, math.inf
        for j in kept:
            if locked[k] and locked[j]:
                continue
            d = pde(tuple(cmed[k]), tuple(cmed[j]))
            if d < bd:
                best, bd = j, d
        if best is not None and bd < limit and not locked[k]:
            target[k] = best
        else:
            kept.append(k)
    new_index = -np.ones(K, np.int64)
    new_index[kept] = np.arange(len(kept))
    remap = new_index[target]
    cls2 = np.where(m, remap[np.maximum(cls, 0)], -1)
    K2 = len(kept)
    med, _ = label_medians(cls2, lab, K2)
    lk = np.asarray(locked)[kept]
    kc = np.zeros(K, bool) if keep_color is None else np.asarray(keep_color)
    for i, k in enumerate(kept):
        if lk[i] or kc[k]:
            med[i] = cmed[k]
    return cls2, med, lk


def _mark_protected(g, protected_lab, tol):
    for r in g.roots():
        m = g.raw_mean(r)
        best = min(protected_lab, key=lambda p: pde(m, p))
        if pde(m, best) <= 0.5 * tol:
            g.protected[r] = best


def vectorize(res, cfg, group_by_color=True):
    """
    Trace the final label map. Returns a list of dicts
    {color, d, area, cls} ordered back-to-front.
    """
    chains, rings = trace_regions(res.labels)
    H, W = res.labels.shape
    segs = []
    for pts, left, right, closed, _, _ in chains:
        # chains running along the image frame stay straight; edges against
        # transparency are smoothed like any other boundary
        straight = (left < 0 or right < 0) and all(
            x == 0 or y == 0 or x == W or y == H for x, y in pts
        )
        segs.append(fit_chain(pts, closed, straight, cfg))
    labels = res.labels
    m = labels >= 0
    area = np.bincount(labels[m], minlength=len(res.region_class))
    out = []
    if group_by_color:
        by_cls = {}
        for r, rl in rings.items():
            by_cls.setdefault(int(res.region_class[r]), []).extend(rl)
        for c, rl in by_cls.items():
            out.append(
                {
                    "cls": c,
                    "color": res.palette_hex[c],
                    "d": rings_to_path(rl, segs),
                    "area": int(res.class_pixels[c]),
                    "regions": int((res.region_class == c).sum()),
                }
            )
    else:
        for r, rl in rings.items():
            c = int(res.region_class[r])
            out.append(
                {"cls": c, "color": res.palette_hex[c], "d": rings_to_path(rl, segs),
                 "area": int(area[r]), "regions": 1}
            )
    out.sort(key=lambda e: -e["area"])
    return out, len(chains)
