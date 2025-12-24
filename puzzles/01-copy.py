"""
Puzzle 01: Copy
==============
This puzzle asks you to implement a copy operation that copies data from one
tensor to another.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle


"""
To begin with, we start to provide a runnable example of TileLang's copy.
The code below shows how to define a 1-D copy kernel using TileLang. We assume
all tensors are stored in the global memory (DRAM) of GPU initially.

01-1: 1-D copy kernel.

Inputs:
    A: [N,]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    B: [N,]  # output tensor

Definition:
    for i in range(N):
        B[i] = A[i]
"""

def ref_copy_1d(A: torch.Tensor, B: torch.Tensor, N: int, dtype: torch.dtype):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert len(A.shape[0] == N)
    assert dtype == A.dtype == B.dtype == torch.float16
    B.copy_(A)

"""
To write a kernel in TileLang, we first define a outer function with some hyper parameters
(like template parameters in many C++ kernel libraryies), e.g. shapes, data types, etc.
In this example, we only need two hyper parameters: N and dtype. This function can be
decorated with `@tilelang.jit` to enable JIT compilation.

Inside this function, we need a kernel function that takes tensors as inputs.
We use `@T.prim_func` to decorate the kernel function. The kernel function should
take tensors as inputs / outputs. All of them are put in the function parameters
list and this function should return void. The body of the kernel function should be
written in TileLang DSL.

We are writing a kernel running on accelerators such as GPU. So we need a kernel launch
configuration. In TileLang, we use `T.Kernel` to launch a kernel. It takes a list of
`blocks` as the number of blocks we want to launch. And an integer `threads` as the
number of threads per block. The kernel function will be launched with `blocks * threads`
number of threads in total.

In the very first step, we just write a serial copy kernel which only launches one thread.
"""
@tilelang.jit
def tl_copy_1d_serial(N: int, dtype: torch.dtype):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype)
    ):
        # The body of the kernel function is written in TileLang DSL.
        # We use T.Kernel to launch a kernel.
        with T.Kernel(1, threads=1) as bx:
            # Here T.copy is a built-in TileOp in TileLang.
            # It will automatically utilize available threads in the block
            # to do efficient memory copy (including auto parallelism and vectorization)
            # As we only launch one thread here, it will be lowered into a serial loop copy
            # with certain bit width vectorization (like 128 bits per copy).
            T.copy(A, B)
    return kernel


def run_copy_1d_serial():
    print("\n=== Copy 1D Serial ===\n")
    N = 1024
    dtype = torch.float16
    test_puzzle(tl_copy_1d_serial, ref_copy_1d, {"N": N, "dtype": dtype})


"""
The above implementation only launches one thread, which is not efficient. Now
we want to launch multiple threads in a single kernel to copy the data in parallel.
As we have T.copy to automatically parallize copying inside one block, we don't need
many modifications to make it work.

Now, just try to change the number of threads in a block to 128/256, and compare
the speedup you get.
"""

@tilelang.jit
def tl_copy_1d_multi_threads(N: int, dtype: torch.dtype):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype)
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_copy_1d_multi_threads():
    print("\n=== Copy 1D Multi-threads ===\n")
    N = 1024*256
    dtype = torch.float16

    test_puzzle(tl_copy_1d_multi_threads, ref_copy_1d, {"N": N, "dtype": dtype})

    bench_puzzle(tl_copy_1d_serial, ref_copy_1d, {"N": N, "dtype": dtype}, bench_name="TL Serial", bench_torch=True)
    bench_puzzle(tl_copy_1d_multi_threads, ref_copy_1d, {"N": N, "dtype": dtype}, bench_name="TL Multi-threads", bench_torch=False)



"""
Finally, we want to parallelize the copy across blocks. We use a BLOCK_N
to denote the elements we need to copy in each block. The rest is similar to the
previous version. We assume that N is divisible by BLOCK_N.

NOTE: You need to deal with the memory access ranges for different blocks. Luckily,
we have `bx` as the block index, so we can compute the start and end indices for each block.
"""

@tilelang.jit
def tl_copy_1d_parallel(N: int, dtype: torch.dtype, BLOCK_N: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype)
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_copy_1d_parallel():
    print("\n=== Copy 1D Parallel ===\n")
    N = 1024*256
    BLOCK_N = 1024
    dtype = torch.float16
    test_puzzle(tl_copy_1d_parallel, ref_copy_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N})
    bench_puzzle(tl_copy_1d_parallel, ref_copy_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N}, bench_name="TL Parallel", bench_torch=True)


if __name__ == "__main__":
    run_copy_1d_serial()
    run_copy_1d_multi_threads()
    run_copy_1d_parallel()