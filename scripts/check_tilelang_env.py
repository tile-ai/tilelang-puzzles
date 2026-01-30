import torch

import tilelang
import tilelang.language as T
from tilelang.env import env

from common.utils import rand_torch_tensor


tilelang.disable_cache()


def run_gemm():
    @tilelang.jit
    def gemm(
        A,
        B,
        block_M: int = 128,
        block_N: int = 128,
        block_K: int = 32,
    ):
        M, N, K = T.const("M, N, K")

        A: T.Tensor[[M, K], T.float16]
        B: T.Tensor[[K, N], T.float16]

        C = T.empty((M, N), T.float16)

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), A.dtype)
            B_shared = T.alloc_shared((block_K, block_N), B.dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_M, by * block_N])
        return C

    A = rand_torch_tensor((2048, 4096), torch.float16)
    B = rand_torch_tensor((4096, 2048), torch.float16)
    C = gemm(A, B)
    C_torch = torch.matmul(A, B)
    print("Check GEMM result: ", torch.allclose(C, C_torch, atol=1e-3))
    print(C.shape)
    print(C_torch.shape)


if __name__ == "__main__":
    print("Installed TileLang version: ", tilelang.__version__)
    print("Installed TileLang Python path: ", tilelang.__path__)
    print("Current CUDA Path: ", env.CUDA_HOME)

    print("*************")
    print("Start torch.utils.collect_env ...")
    torch.utils.collect_env.main()

    print("*************")
    print("Start compiling & running a simple GEMM kernel")
    run_gemm()
