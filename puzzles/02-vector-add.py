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

from utils import test_puzzle, bench_puzzle

"""
Vector addition is our first step towards computation. Tilelang provides basic
arithmetic operations like add, sub, mul, div, etc. But these operations are
element-wise (They are not TileOps like T.copy, since they are executed in CUDA Core).

So we need a loop abstraction to iterate over elements in the tensor. Inside the loop body,
we can perform whatever computation we want.

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

def ref_add_1d(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, dtype: torch.dtype):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0] == N
    assert dtype == A.dtype == B.dtype == C.dtype == torch.float16
    torch.add(input=A, other=B, out=C)


@tilelang.jit
def tl_add_1d(N: int, dtype: torch.dtype, BLOCK_N: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype),
        C: T.Buffer((N,), dtype)
    ):
        with T.Kernel(N // BLOCK_N, threads=256) as bx:
            base_idx = bx * BLOCK_N
            for i in T.Parallel(BLOCK_N):
                C[base_idx + i] = A[base_idx + i] + B[base_idx + i]

    return kernel


def run_add_1d():
    print("\n=== Vector Add 1D ===\n")
    N = 1024*256
    BLOCK_N = 1024
    dtype = torch.float16
    test_puzzle(tl_add_1d, ref_add_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N})


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


def ref_mul_relu_1d(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, dtype: torch.dtype):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0] == N
    assert dtype == A.dtype == B.dtype == torch.float16
    torch.mul(input=A, other=B, out=C)
    C.relu_()


@tilelang.jit
def tl_mul_relu_1d(N: int, dtype: torch.dtype, BLOCK_N: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype),
        C: T.Buffer((N,), dtype)
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_mul_relu_1d():
    print("\n=== Vector Multiplication with ReLU 1D ===\n")
    N = 1024*256
    BLOCK_N = 1024
    dtype = torch.float16
    test_puzzle(tl_mul_relu_1d, ref_mul_relu_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N})


"""
NOTE: This section needs some understanding of GPU memory hierarchy and basic CUDA programming knowledge.

We can go further in the above example. Here we introduce an common optimizations when writing
kernels. If you have some experience with CUDA or other GPU programming, you are likely to know
that there exists a memory hierarchy on GPU.

Commonly, there are three levels of memory: global memory (DRAM), shared memory (shared), a
nd register (register). Registers are the fastest memory, but also the smallest. In CUDA,
the registers are allocated when you declare a local variable in a kernel.

Our above implementation directly loads data from A, B and stores the result to C, while
A, B, C are all passed in as global memory pointers. This is inefficient because it requires
to access global memory for every single element. One can use `print_source_code()` to see
the generated CUDA code.

Here we consider using registers to optimize the kernel. The key idea is that we can copy multiple
data from/to registers at once. For example, CUDA usually use `ldg128` to load data from global
memory to registers, which loads 128bits at once. This theoretically reduces the number of memory
accesses by 4x.

And in our fused kernel example, the intermediate results of A * B can also be stored in registers.
And when we do ReLU, we directly read from registers instead of global memory. (Maybe we don't need
to explicitly do this. It can be optimized by NVCC in a Common-Subexpr-Elimination (CSE) optimization)
"""

"""
TileLang explicitly exposes these memory levels to users. You can use `T.alloc_fragment`
to allocate a fragment of registers. Note that when you write CUDA, registers are thread-local.
So when you write programs, you usually need to handle some logics to make sure each thread load
certain part of the data into registers. But in TileLang, you don't need to do such mappings.
A fragment is an abstraction of registers in all threads in a block. We can manipulate this fragment
in a unified way as we do to a T.Buffer.
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_mul_relu_1d_mem(N: int, dtype: torch.dtype, BLOCK_N: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype),
        C: T.Buffer((N,), dtype)
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_mul_relu_1d_mem():
    print("\n=== Vector Multiplication with ReLU 1D (Memory Optimized) ===\n")
    N = 1024*4096
    BLOCK_N = 1024
    dtype = torch.float16

    print("Naive TL Implementation: ")
    tl_mul_relu_kernel = tl_mul_relu_1d(N, dtype, BLOCK_N)
    tl_mul_relu_kernel.print_source_code()

    print("Optimized Version")
    tl_mul_relu_kernel_opt = tl_mul_relu_1d_mem(N, dtype, BLOCK_N)
    tl_mul_relu_kernel_opt.print_source_code()

    test_puzzle(tl_mul_relu_1d_mem, ref_mul_relu_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N})
    bench_puzzle(tl_mul_relu_1d, ref_mul_relu_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N}, bench_name="TL Naive", bench_torch=True)
    bench_puzzle(tl_mul_relu_1d_mem, ref_mul_relu_1d, {"N": N, "dtype": dtype}, {"BLOCK_N": BLOCK_N}, bench_name="TL OPT", bench_torch=False)




if __name__ == "__main__":
    run_add_1d()
    run_mul_relu_1d()
    run_mul_relu_1d_mem()