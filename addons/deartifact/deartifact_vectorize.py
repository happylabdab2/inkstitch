#!/usr/bin/env python3
"""
Inkscape extension: De-artifact & Vectorize.

Takes the selected bitmap(s), reconstructs the most likely original flat
colour regions (compression-artifact aware, topology preserving) and writes
either clean SVG paths, a cleaned bitmap for Trace Bitmap, or both.
"""

import base64
import io
import os
import sys
import time
import urllib.parse
import urllib.request

import inkex
from inkex import Group, PathElement, Transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

import deartifact_engine as E  # noqa: E402


class DeArtifactVectorize(inkex.EffectExtension):
    def add_arguments(self, pars):
        pars.add_argument("--tab")
        # restoration
        pars.add_argument("--profile", default="balanced")
        pars.add_argument("--tolerance", type=float, default=8.0)
        pars.add_argument("--edge_preservation", type=int, default=75)
        pars.add_argument("--smooth_iterations", type=int, default=2)
        pars.add_argument("--superpixel_size", type=int, default=0)
        # colours
        pars.add_argument("--target_colors", type=int, default=0)
        pars.add_argument("--protected_colors", default="")
        # islands
        pars.add_argument("--island_sensitivity", type=int, default=4)
        pars.add_argument("--island_max_area", type=int, default=0)
        pars.add_argument("--protect_holes", type=inkex.Boolean, default=True)
        # output
        pars.add_argument("--output_mode", default="vector")
        pars.add_argument("--group_by_color", type=inkex.Boolean, default=True)
        pars.add_argument("--trace_tolerance", type=float, default=1.0)
        pars.add_argument("--corner_angle", type=float, default=55.0)
        pars.add_argument("--smoothness", type=float, default=1.0)
        pars.add_argument("--seam_stroke", type=float, default=0.4)
        pars.add_argument("--original", default="hide")
        pars.add_argument("--max_megapixels", type=float, default=2.0)
        pars.add_argument("--alpha_threshold", type=int, default=128)
        pars.add_argument("--report", type=inkex.Boolean, default=True)

    # ------------------------------------------------------------------
    def effect(self):
        images = []
        for el in self.svg.selection.values():
            if isinstance(el, inkex.Image):
                images.append(el)
            else:
                images.extend(el.descendants().filter(inkex.Image))
        if not images and not self.svg.selection:
            # nothing selected: process every bitmap in the document
            images = list(self.svg.descendants().filter(inkex.Image))
        if not images:
            raise inkex.AbortExtension(
                "No bitmap images found. Import or paste an image, select it, and run again."
            )
        reports = []
        for img in images:
            reports.append(self.process(img))
        if self.options.report:
            self.msg("\n\n".join(r for r in reports if r))

    # ------------------------------------------------------------------
    def config(self):
        o = self.options
        protected = [c for c in o.protected_colors.replace(";", ",").replace(" ", ",").split(",") if c.strip()]
        for c in protected:
            try:
                E.hex_to_lab(c)
            except ValueError:
                raise inkex.AbortExtension("Protected colour '%s' is not a #rrggbb value." % c)
        return E.Config(
            profile=o.profile,
            tolerance=o.tolerance,
            edge_preservation=o.edge_preservation / 100.0,
            smooth_iterations=o.smooth_iterations,
            superpixel_size=o.superpixel_size,
            target_colors=o.target_colors,
            protected_colors=protected,
            island_sensitivity=o.island_sensitivity,
            island_max_area=o.island_max_area,
            protect_holes=o.protect_holes,
            trace_tolerance=o.trace_tolerance,
            corner_angle=o.corner_angle,
            smoothness=o.smoothness,
            alpha_threshold=o.alpha_threshold,
        )

    def load_pixels(self, img):
        href = img.get("xlink:href") or img.get("href") or ""
        try:
            if href.startswith("data:"):
                meta, data = href.split(",", 1)
                if "svg" in meta:
                    raise inkex.AbortExtension("Image %s is an embedded SVG, not a bitmap." % img.get_id())
                raw = base64.b64decode(data) if ";base64" in meta else urllib.parse.unquote_to_bytes(data)
                pil = PILImage.open(io.BytesIO(raw))
            else:
                path = href
                if path.startswith("file:"):
                    path = urllib.request.url2pathname(urllib.parse.urlparse(path).path)
                path = self.absolute_href(urllib.parse.unquote(path))
                if not os.path.isfile(path):
                    raise inkex.AbortExtension("Linked image not found: %s" % path)
                pil = PILImage.open(path)
            pil.load()
        except inkex.AbortExtension:
            raise
        except Exception as exc:  # PIL errors
            raise inkex.AbortExtension("Could not read image %s: %s" % (img.get_id(), exc))
        return pil.convert("RGBA")

    @staticmethod
    def placement(img, Wp, Hp):
        """Map image pixel space -> user space of the image's parent."""
        x = img.left
        y = img.top
        w = img.width if img.get("width") else float(Wp)
        h = img.height if img.get("height") else float(Hp)
        sx, sy = w / Wp, h / Hp
        par = (img.get("preserveAspectRatio") or "xMidYMid meet").split()
        if par and par[0] != "none":
            s = max(sx, sy) if "slice" in par else min(sx, sy)
            al = par[0]
            fx = 0.0 if "xMin" in al else (1.0 if "xMax" in al else 0.5)
            fy = 0.0 if "YMin" in al else (1.0 if "YMax" in al else 0.5)
            x += (w - Wp * s) * fx
            y += (h - Hp * s) * fy
            sx = sy = s
        return x, y, sx, sy

    # ------------------------------------------------------------------
    def process(self, img):
        o = self.options
        t0 = time.time()
        cfg = self.config()
        pil = self.load_pixels(img)
        Wp, Hp = pil.size
        scale = 1.0
        max_px = max(0.05, o.max_megapixels) * 1e6
        if Wp * Hp > max_px:
            scale = (max_px / (Wp * Hp)) ** 0.5
            pil = pil.resize((max(1, int(Wp * scale)), max(1, int(Hp * scale))), PILImage.LANCZOS)
        rgba = np.asarray(pil, dtype=np.uint8)
        Hw, Ww = rgba.shape[:2]
        if not (rgba[..., 3] >= cfg.alpha_threshold).any():
            raise inkex.AbortExtension("Image %s is fully transparent." % img.get_id())

        res = E.deartifact(rgba, cfg)
        t1 = time.time()

        x, y, sx, sy = self.placement(img, Wp, Hp)
        # working pixel -> user units
        px_to_user = (
            img.transform
            @ Transform(translate=(x, y))
            @ Transform(scale=(sx * Wp / Ww, sy * Hp / Hw))
        )
        parent = img.getparent()
        idx = parent.index(img)
        insert_at = idx + 1
        name = img.label or img.get_id()

        if o.output_mode in ("bitmap", "both"):
            clean = PILImage.fromarray(res.cleaned_rgba, "RGBA")
            if scale != 1.0:
                clean = clean.resize((Wp, Hp), PILImage.NEAREST)
            buf = io.BytesIO()
            clean.save(buf, "PNG", optimize=True)
            bm = inkex.Image()
            for attr in ("x", "y", "width", "height", "preserveAspectRatio", "transform"):
                if img.get(attr) is not None:
                    bm.set(attr, img.get(attr))
            if img.get("width") is None:
                bm.set("width", str(Wp))
                bm.set("height", str(Hp))
            bm.set("xlink:href", "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
            bm.style["image-rendering"] = "optimizeSpeed"
            bm.label = "%s (de-artifacted bitmap, %d colours)" % (name, len(res.palette_hex))
            if o.output_mode == "both":
                bm.style["display"] = "none"  # keep for Trace Bitmap; don't show under the curves
            parent.insert(insert_at, bm)
            insert_at += 1

        n_paths = n_nodes = n_chains = 0
        if o.output_mode in ("vector", "both"):
            items, n_chains = E.vectorize(res, cfg, group_by_color=o.group_by_color)
            group = Group()
            group.label = "%s (vector, %d colours)" % (name, len(res.palette_hex))
            group.transform = px_to_user
            for it in items:
                if not it["d"]:
                    continue
                p = PathElement()
                p.set("d", it["d"])
                style = {"fill": it["color"], "fill-rule": "evenodd"}
                if o.seam_stroke > 0:
                    style.update(
                        {
                            "stroke": it["color"],
                            "stroke-width": "%g" % o.seam_stroke,
                            "stroke-linejoin": "round",
                        }
                    )
                else:
                    style["stroke"] = "none"
                p.style = inkex.Style(style)
                if o.group_by_color:
                    p.label = "%s (%d region%s)" % (it["color"], it["regions"], "" if it["regions"] == 1 else "s")
                else:
                    p.label = it["color"]
                group.append(p)
                n_paths += 1
                n_nodes += it["d"].count("C") + it["d"].count("L") + it["d"].count("M")
            parent.insert(insert_at, group)

        if o.original == "hide":
            img.style["display"] = "none"
        elif o.original == "delete":
            img.delete()

        if not o.report:
            return ""
        info = res.info
        total = max(1, int(res.class_pixels.sum()))
        order = np.argsort(-res.class_pixels)
        pal_lines = [
            "   %s  %5.1f%%%s" % (res.palette_hex[k], 100.0 * res.class_pixels[k] / total, "  (protected)" if res.locked[k] else "")
            for k in order
        ]
        lines = [
            "De-artifact & Vectorize: %s" % name,
            "  Image: %d x %d px%s" % (Wp, Hp, "" if scale == 1.0 else "  (processed at %d x %d)" % (Ww, Hw)),
            "  Analysis: %d unique colours, noise %.2f dE, 8x8 blocking ratio %.2f%s"
            % (info["unique_colors"], info["noise_estimate"], info["blockiness"],
               " (JPEG-like)" if info["jpeg_like"] else ""),
            "  Regions: %d initial -> %d merged in graph, %d AA strips split, %d halos and %d islands absorbed, %d small shapes protected"
            % (info["initial_regions"], info["merges"], info["transition_strips"],
               info.get("halo_strips", 0), info["islands_absorbed"], info["islands_protected"]),
            "  Result: %d palette colours, %d regions" % (len(res.palette_hex), info["final_regions"]),
        ]
        if o.output_mode in ("vector", "both"):
            lines.append("  Vector: %d paths, %d shared boundary chains, ~%d nodes" % (n_paths, n_chains, n_nodes))
        lines.append("  Time: %.1fs analysis/segmentation, %.1fs total" % (t1 - t0, time.time() - t0))
        lines.append("  Palette:")
        lines.extend(pal_lines)
        return "\n".join(lines)


if __name__ == "__main__":
    DeArtifactVectorize().run()
