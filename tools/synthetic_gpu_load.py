"""Oscillating synthetic GPU load, for validating telemetry against a varying signal.

This exists to close one specific gate: an energy-counter cross-check run at idle
proves nothing, because on a constant power signal every integration method
agrees. Validating the counter requires power that actually moves, on a period
comparable to the sensor's refresh interval.

Deliberately dependency-free. It talks to the CUDA driver API (libcuda.so.1,
shipped with the driver) through ctypes and JITs a hand-written PTX kernel, so it
needs no CUDA toolkit, no nvcc, and no torch/cupy. That matters because this has
to run on a rented H100 as easily as on a local dev box.

Not a benchmark workload. Component 3 (the vLLM load generator) drives realistic
inference traffic; this only makes the power rail swing on a known schedule.

Usage:
    python tools/synthetic_gpu_load.py --duration 40 --period 2.0 --duty 0.5
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

# d = a * b + c with a=f1, b=0.5, c=0.5 converges toward 1.0, so the accumulator
# stays finite -- an inf/NaN accumulator can change how the FPU draws power.
PTX = b"""
.version 6.0
.target sm_52
.address_size 64

.visible .entry burn(
    .param .u64 burn_param_0,
    .param .u32 burn_param_1
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<5>;

    ld.param.u64        %rd1, [burn_param_0];
    ld.param.u32        %r1,  [burn_param_1];
    cvta.to.global.u64  %rd2, %rd1;

    mov.u32     %r2, %ctaid.x;
    mov.u32     %r3, %ntid.x;
    mov.u32     %r4, %tid.x;
    mad.lo.s32  %r5, %r2, %r3, %r4;

    mov.f32     %f1, 0f3F800000;
    mov.f32     %f2, 0f3F000000;
    mov.f32     %f3, 0f3F000000;
    mov.u32     %r6, 0;

LOOP:
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    fma.rn.f32  %f1, %f1, %f2, %f3;
    add.s32     %r6, %r6, 1;
    setp.lt.u32 %p1, %r6, %r1;
    @%p1 bra    LOOP;

    mul.wide.u32   %rd3, %r5, 4;
    add.s64        %rd4, %rd2, %rd3;
    st.global.f32  [%rd4], %f1;
    ret;
}
"""


class CudaError(RuntimeError):
    pass


class Cuda:
    """Minimal CUDA driver-API wrapper. Compute only; touches no clock/power state."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:  # pragma: no cover - environment dependent
            raise CudaError(f"libcuda.so.1 not loadable: {exc}") from exc
        self.ctx = None
        self.dptr = None

    def check(self, code: int, what: str) -> None:
        if code != 0:
            msg = ctypes.c_char_p()
            try:
                self.lib.cuGetErrorString(code, ctypes.byref(msg))
                detail = msg.value.decode() if msg.value else "?"
            except Exception:  # pragma: no cover
                detail = "?"
            raise CudaError(f"{what} failed: CUresult={code} ({detail})")

    def setup(self, device: int, threads: int, blocks: int) -> None:
        self.check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self.check(
            self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate"
        )
        self.ctx = ctx
        mod = ctypes.c_void_p()
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), PTX), "cuModuleLoadData")
        fn = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"burn"),
            "cuModuleGetFunction",
        )
        self.fn = fn
        nbytes = threads * blocks * 4
        dptr = ctypes.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(dptr), nbytes), "cuMemAlloc")
        self.dptr = dptr
        self.threads = threads
        self.blocks = blocks

    def burn(self, iters: int) -> None:
        out = ctypes.c_uint64(self.dptr.value)
        n = ctypes.c_uint32(iters)
        params = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n), ctypes.c_void_p),
        )
        self.check(
            self.lib.cuLaunchKernel(
                self.fn,
                self.blocks, 1, 1,
                self.threads, 1, 1,
                0, None, params, None,
            ),
            "cuLaunchKernel",
        )
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        if self.dptr is not None:
            self.lib.cuMemFree_v2(self.dptr)
            self.dptr = None
        if self.ctx is not None:
            self.lib.cuCtxDestroy_v2(self.ctx)
            self.ctx = None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--duration", type=float, default=40.0, help="total seconds")
    p.add_argument("--period", type=float, default=2.0, help="burst+idle period (s)")
    p.add_argument("--duty", type=float, default=0.5, help="fraction of period burning")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--blocks", type=int, default=512)
    p.add_argument("--chunk-iters", type=int, default=200_000,
                   help="loop iterations per launch; tune so one launch is ~50ms")
    args = p.parse_args(argv)

    cuda = Cuda()
    try:
        cuda.setup(args.device, args.threads, args.blocks)

        # Calibrate: time one launch so bursts can be built from whole launches
        # rather than interrupted mid-kernel.
        t = time.monotonic()
        cuda.burn(args.chunk_iters)
        per_launch = time.monotonic() - t
        print(
            f"launch: {per_launch * 1000:.1f} ms "
            f"({args.blocks}x{args.threads} threads, {args.chunk_iters} iters)",
            flush=True,
        )

        burn_s = args.period * args.duty
        idle_s = args.period - burn_s
        end = time.monotonic() + args.duration
        cycles = 0
        while time.monotonic() < end:
            t_burn_end = min(time.monotonic() + burn_s, end)
            while time.monotonic() < t_burn_end:
                cuda.burn(args.chunk_iters)
            if time.monotonic() >= end:
                break
            time.sleep(min(idle_s, max(0.0, end - time.monotonic())))
            cycles += 1
        print(f"done: {cycles} cycles over {args.duration:.0f}s", flush=True)
    except CudaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        cuda.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
