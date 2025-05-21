from math import ceildiv
from pathlib import Path

from gpu.host import DeviceContext
from kernels.barrier import simple_kernel, foo, bar
from memory import UnsafePointer


fn main() -> None:
    alias block_dim = 32
    alias type = DType.float32
    alias M = 16384
    alias N = 4096
    alias K = 4096
    alias alpha = 2.0
    alias beta = 1.0

    y = foo(999999999)
    print(y)
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function[
                simple_kernel, dump_asm = Path("./barrier.ptx")
            ](
                grid_dim=(ceildiv(N, block_dim), ceildiv(M, block_dim)),
                block_dim=(block_dim, block_dim),
            )
    except:
        pass
