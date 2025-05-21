from math import ceildiv, fma
from memory import UnsafePointer
from memory import stack_allocation
from pathlib import Path
from sys import simdwidthof

import compiler_internal as compiler
from gpu import barrier, global_idx, thread_idx, block_idx
from gpu.memory import AddressSpace
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from utils.numerics import get_accum_type


fn bar(x: Int) -> Int:
    return UnsafePointer[Int].alloc(42)[x]


fn foo(x: Int) -> Int:
    return bar(x)


fn simple_kernel() -> None:
    """Kernel that just has a barrier."""
    barrier()


@compiler.register("simple_kernel")
struct SimpleKernel:
    """Simple kernel containing a barrier."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](c: MutableInputTensor, ctx: DeviceContextPtr) raises -> None:
        alias block_dim = 32
        M = c.dim_size(0)
        N = c.dim_size(1)

        gpu_ctx = ctx.get_device_context()

        gpu_ctx.enqueue_function[
            simple_kernel, dump_asm = Path("./barrier.ptx")
        ](
            grid_dim=(ceildiv(N, block_dim), ceildiv(M, block_dim)),
            block_dim=(block_dim, block_dim),
        )
