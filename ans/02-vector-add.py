"""
Puzzle 02: Vector Add
==============
This puzzle asks you to implement a vector addition operation.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Vector addition is our first step towards computation. Tilelang provides basic arithmetic
operations like add, sub, mul, div, etc. But these operations are element-wise (They are not
TileOps like T.copy). So we need a loop abstraction to iterate over elements in the tensor.
Inside the loop body, we can perform whatever computation we want.

02-1: 1-D vector addition.

Inputs:
    A: [N,]  # input tensor
    B: [N,]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    C: [N,]  # output tensor

Definition:
    for i in range(N):
        C[i] = A[i] + B[i]
"""


def ref_add_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return A + B


@tilelang.jit
def tl_add_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            C[base_idx + i] = A[base_idx + i] + B[base_idx + i]

    return C


def run_add_1d():
    print("\n=== Vector Add 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_add_1d, ref_add_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
We can fuse more elementwise operations into this kernel.
Now that's do an element-wise multiplication with a ReLU activation.

HINT: We can use T.if_then_else(cond, true_value, false_value) to implement conditional logic.

02-2: 1-D vector multiplication with ReLU activation

Inputs:
    A: [N,]  # input tensor
    B: [N,]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    C: [N,]  # output tensor

Definition:
    for i in range(N):
        C[i] = max(0, A[i] * B[i])
"""


def ref_mul_relu_1d(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0]
    assert A.dtype == B.dtype == torch.float16
    return (A * B).relu_()


@tilelang.jit
def tl_mul_relu_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        with T.Kernel(N // BLOCK_N, threads=256) as bx:
            base_idx = bx * BLOCK_N
            for i in T.Parallel(BLOCK_N):
                C[base_idx + i] = T.if_then_else(
                    A[base_idx + i] * B[base_idx + i] > 0,
                    A[base_idx + i] * B[base_idx + i],
                    0,
                )

    return C


def run_mul_relu_1d():
    print("\n=== Vector Multiplication with ReLU 1D ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_mul_relu_1d, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})


"""
NOTE: This section needs some understanding of GPU memory hierarchy and basic CUDA
programming knowledge.

We can further optimize the previous example. Here, we introduce a common optimization technique
used in kernel programming. If you have experience with CUDA or other GPU programming frameworks,
you are likely aware of the memory hierarchy on GPUs.

Typically, there are three main levels of memory: global memory (DRAM), shared memory, and
registers. Registers are the fastest but also the smallest form of memory. In CUDA, registers are
allocated when you declare local variables within a kernel.

Our previous implementation loads data directly from A and B and stores the result to C, where A, B,
and C are all passed as global memory pointers. This is inefficient because it requires accessing
global memory for every single element. You can use print_source_code() to inspect the generated
CUDA code.

Here, we consider using registers to optimize the kernel. The key idea is to copy multiple data
elements between registers and global memory in a single operation. For example, CUDA often uses
ldg128 to load 128 bits of data from global memory into registers at once, which can theoretically
reduce the number of memory accesses by 4x.

In our fused kernel example, intermediate results from A * B can also be stored in registers. When
applying the ReLU operation, we can read directly from registers instead of global memory. (In
practice, this may not need to be done explicitly—it can often be optimized automatically by
NVCC through common subexpression elimination, or CSE.)
"""

"""
TileLang explicitly exposes these memory levels to users. You can use `T.alloc_fragment`
to allocate a fragment of registers. Note that when you write CUDA, registers are thread-local.
So when you write programs, you usually need to handle some logics to make sure each thread load
certain part of the data into registers. But in TileLang, you don't need to do such mappings.
A fragment is an abstraction of registers in all threads in a block. We can manipulate this
fragment in a unified way as we do to a T.Buffer.
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_1d_mem(A, B, BLOCK_N: int):
    N = T.const("N")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, threads=256) as bx:
        base_idx = bx * BLOCK_N
        A_local = T.alloc_fragment((BLOCK_N), dtype)
        B_local = T.alloc_fragment((BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_N), dtype)

        T.copy(A[base_idx], A_local)
        T.copy(B[base_idx], B_local)

        for i in T.Parallel(BLOCK_N):
            C_local[i] = A_local[i] * B_local[i]
            C_local[i] = T.if_then_else(C_local[i] > 0, C_local[i], 0)

        T.copy(C_local, C[base_idx])

    return C


def run_mul_relu_1d_mem():
    print("\n=== Vector Multiplication with ReLU 1D (Memory Optimized) ===\n")
    N = 1024 * 4096
    BLOCK_N = 1024

    print("Naive TL Implementation: ")
    tl_mul_relu_kernel = tl_mul_relu_1d.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel.print_source_code()

    print("Optimized Version")
    tl_mul_relu_kernel_opt = tl_mul_relu_1d_mem.compile(N=N, BLOCK_N=BLOCK_N)
    tl_mul_relu_kernel_opt.print_source_code()

    test_puzzle(tl_mul_relu_1d_mem, ref_mul_relu_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_mul_relu_1d,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Naive",
        bench_torch=True,
    )
    bench_puzzle(
        tl_mul_relu_1d_mem,
        ref_mul_relu_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL OPT",
        bench_torch=False,
    )


if __name__ == "__main__":
    run_add_1d()
    run_mul_relu_1d()
    run_mul_relu_1d_mem()
