"""
Puzzle 09: Convolution
==============
Convolution is another fundamental computation pattern in deep learning operators.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle

"""
Convolution uses a sliding window approach to compute over a input tensors. The main characteristics of convolution is that it has strong data reuse patterns and requires careful memory access optimization. But with TileLang, we can ignore most of these details and focus on the logic.

In this puzzle, we remove the "channel (C)" dimension to simplify the problem. We first look at the 1D convolution case, then extend to 2D. And we will learn how to use shared memory of GPU in this chapter.

08-1: 1D Convolution.

Inputs:
    X: [N, L]  # input tensor
    K: [KL,]  # kernel tensor
    N: int   # batch size dimension. 1 <= N <= 64
    H: int   # length dimension. 1 <= H <= 1024
    KL: int  # kernel height. 1 <= KH <= 32
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    O: [N, L]  # output tensor

Definition:
    for i in range(N):
        for j in range(L):
            O[i, j] = 0
            for k in range(KL):
                if j + k < L:  # boundary check
                    O[i, j] += X[i, j + k] * K[k]
"""


"""
We can first consider a naive implementation. We can parallelize the outer loop over `N` and `L` to different blocks with `T.Kernel`.
For the loop iterating `BLOCK_L`, we can use a serial implementation for now. Be careful that the data dependency in the convolution.
"""


def ref_conv1d(X: torch.Tensor, K: torch.Tensor, O: torch.Tensor, N: int, L: int, KL: int, dtype: torch.dtype):
    assert len(X.shape) == 2
    assert len(K.shape) == 1
    assert len(O.shape) == 2
    assert X.shape[0] == O.shape[0] == N
    assert X.shape[1] == O.shape[1] == L
    assert K.shape[0] == KL
    assert dtype == X.dtype == K.dtype == O.dtype == torch.float32

    # for i in range(N):
    #     for j in range(L):
    #         O[i, j] = 0
    #         for k in range(KL):
    #             if j + k < L:  # boundary check
    #                 O[i, j] += X[i, j + k] * K[k]

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    O.copy_(torch.conv1d(
        input=X_padded,
        weight=K.view(1, 1, KL),
    ).view(N, L))


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_naive(N: int, L: int, KL: int, dtype: torch.dtype, BLOCK_N: int, BLOCK_L: int):
    @T.prim_func
    def kernel(
        X: T.Buffer((N, L), dtype),
        K: T.Buffer((KL), dtype),
        O: T.Buffer((N, L), dtype),
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_conv1d_naive():
    print("\n=== Convolution 1D Naive ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    dtype = torch.float32
    test_puzzle(tl_conv1d_naive, ref_conv1d, {"N": N, "L": L, "KL": KL, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L})



"""
The naive implementation of Conv 1D works but it is not efficient. Remember that we mentioned Tensor Core and `T.gemm` in previous puzzle? Actually, we can also convert the convolution problem to a GEMM problem through a transformation called `im2col`. The idea is to transform the convolution into a matrix multiplication where each row of the input matrix corresponds to a local patch of the input tensor, and the kernel is reshaped into a matrix. This allows us to leverage highly optimized GEMM implementations.

To present GEMM from degenerating to GEMV, we need to introduce an output channel dimension F.

08-2: 1D Convolution with multiple output channels.

Inputs:
    X: [N, L]  # input tensor
    K: [KL, F]  # kernel tensor
    N: int   # batch size dimension. 1 <= N <= 64
    H: int   # length dimension. 1 <= H <= 1024
    KL: int  # kernel height. 1 <= KH <= 32
    F: int   # filter channels. 32 <= F <= 128
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    O: [N, L, F]  # output tensor

Definition:
    for i in range(N):
        for j in range(L):
            for f in range(F):
                O[i, j, f] = 0
                for k in range(KL):
                    if j + k < L:  # boundary check
                        O[i, j, f] += X[i, j + k] * K[k, f]
"""

def ref_conv1d_multi_outchannel(X: torch.Tensor, K: torch.Tensor, O: torch.Tensor, N: int, L: int, KL: int, F: int, dtype: torch.dtype):
    assert len(X.shape) == 2
    assert len(K.shape) == 2
    assert len(O.shape) == 3
    assert X.shape[0] == O.shape[0] == N
    assert X.shape[1] == O.shape[1] == L
    assert O.shape[2] == K.shape[1] == F
    assert K.shape[0] == KL
    assert dtype == X.dtype == K.dtype == O.dtype == torch.float32

    # for i in range(N):
    #     for j in range(L):
    #         for f in range(F):
    #             O[i, j, f] = 0
    #             for k in range(KL):
    #                 if j + k < L:  # boundary check
    #                     O[i, j, f] += X[i, j + k] * K[k, f]

    padding_size = KL - 1
    X_padded = torch.nn.functional.pad(X.view(N, 1, L), (0, padding_size))

    O.copy_(torch.conv1d(
        input=X_padded,
        weight=K.permute(1, 0).view(F, 1, KL),
    ).permute(0, 2, 1).contiguous())



@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_multi_outchannel(N: int, L: int, KL: int, F: int, dtype: torch.dtype, BLOCK_N: int, BLOCK_L: int):
    @T.prim_func
    def kernel(
        X: T.Buffer((N, L), dtype),
        K: T.Buffer((KL, F), dtype),
        O: T.Buffer((N, L, F), dtype),
    ):
        # TODO: Implement this function
        pass

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_conv1d_img2col(N: int, L: int, KL: int, F: int, dtype: torch.dtype, BLOCK_N: int, BLOCK_L: int):
    @T.prim_func
    def kernel(
        X: T.Buffer((N, L), dtype),
        K: T.Buffer((KL, F), dtype),
        O: T.Buffer((N, L, F), dtype),
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_conv1d_img2col():
    print("\n=== Convolution 1D Img2Col ===\n")
    N = 128
    L = 128
    BLOCK_N = 16
    BLOCK_L = 32
    KL = 32
    F = 32
    dtype = torch.float32
    test_puzzle(tl_conv1d_multi_outchannel, ref_conv1d_multi_outchannel, {"N": N, "L": L, "KL": KL, "F": F, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L})
    test_puzzle(tl_conv1d_img2col, ref_conv1d_multi_outchannel, {"N": N, "L": L, "KL": KL, "F": F, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L})
    bench_puzzle(tl_conv1d_multi_outchannel, ref_conv1d_multi_outchannel, {"N": N, "L": L, "KL": KL, "F": F, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L}, bench_torch=True, bench_name="Conv1D Multi OutChannel Naive")
    bench_puzzle(tl_conv1d_img2col, ref_conv1d_multi_outchannel, {"N": N, "L": L, "KL": KL, "F": F, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_L": BLOCK_L}, bench_torch=False, bench_name="Conv1D Img2Col")



if __name__ == "__main__":
    # run_conv1d_naive()
    run_conv1d_img2col()