"""Optional WebGPU kernels for workloads that benefit from parallel execution."""

import os
import struct
import threading

from .compute import get_compute_backend


_SPLIT_THRESHOLD = 512
_MAX_SPLITS = 4_194_240
_MAX_SEGMENT_COMPONENT = 10_000.0
_RESOURCE_LOCK = threading.Lock()
_RESOURCES = None
_RESOURCE_ATTEMPTED = False
_DISABLED = False

_SHADER = """
struct Params {
    split_count: u32,
    _padding: u32,
    delta: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> offsets: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= params.split_count {
        return;
    }
    let fraction = f32(index + 1u) / f32(params.split_count + 1u);
    offsets[index] = params.delta * fraction;
}
"""


def split_segment_webgpu(a, b, segments: int):
    """Return evenly spaced interior coordinates, or None to use the CPU path."""
    if _DISABLED:
        return None

    backend = get_compute_backend()
    if backend == "cpu" or backend == "threaded" or segments <= _SPLIT_THRESHOLD or segments > _MAX_SPLITS:
        return None

    start_x, start_y = _coordinates(a)
    end_x, end_y = _coordinates(b)
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    if max(abs(delta_x), abs(delta_y)) > _MAX_SEGMENT_COMPONENT:
        return None

    try:
        resources = _get_resources()
        if resources is None:
            return None
        device, pipeline, wgpu = resources

        split_count = segments - 1
        params = device.create_buffer_with_data(
            label="Ink/Stitch split parameters",
            data=struct.pack("<II2f", split_count, 0, delta_x, delta_y),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        output = device.create_buffer(
            label="Ink/Stitch split offsets",
            size=split_count * 8,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params, "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": output, "offset": 0, "size": split_count * 8}},
            ],
        )

        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((split_count + 63) // 64)
        compute_pass.end()
        device.queue.submit([encoder.finish()])

        values = struct.unpack(f"<{split_count * 2}f", device.queue.read_buffer(output))
        return [
            (start_x + values[index * 2], start_y + values[index * 2 + 1])
            for index in range(split_count)
        ]
    except Exception:
        _disable_backend()
        return None


def _coordinates(point):
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def _get_resources():
    global _RESOURCES, _RESOURCE_ATTEMPTED
    if _RESOURCE_ATTEMPTED:
        return _RESOURCES

    with _RESOURCE_LOCK:
        if _RESOURCE_ATTEMPTED:
            return _RESOURCES

        try:
            import wgpu

            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter is not None:
                device = adapter.request_device_sync()
                shader = device.create_shader_module(code=_SHADER)
                pipeline = device.create_compute_pipeline(
                    layout="auto",
                    compute={"module": shader, "entry_point": "main"},
                )
                _RESOURCES = device, pipeline, wgpu
        except Exception:
            _RESOURCES = None
        _RESOURCE_ATTEMPTED = True
        return _RESOURCES


def _disable_backend():
    global _DISABLED
    with _RESOURCE_LOCK:
        _DISABLED = True