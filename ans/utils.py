"""
Utilities for TileLang puzzles.
"""

import tilelang

# We disable tilelang cache for turtorial.
tilelang.disable_cache()

from tilelang.jit.kernel import JITKernel
from tilelang.engine.param import KernelParam

import torch

# torch.set_printoptions(profile="full")


def _tvm_ffi_dtype_to_torch_dtype(ffi_dtype) -> torch.dtype:
    if ffi_dtype == "float16":
        return torch.float16
    elif ffi_dtype == "float32":
        return torch.float32
    elif ffi_dtype == "uint8":
        return torch.uint8
    elif ffi_dtype == "int32":
        return torch.int32
    elif ffi_dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {ffi_dtype}")



def _torch_tensor_materialize(params: list[KernelParam]):
    inputs_in_torch_tensors: list[torch.Tensor] = []
    output_torch: torch.Tensor = None
    output_tl: torch.Tensor = None

    for idx, tl_param in enumerate(params):
        shape = tl_param.shape
        dtype = tl_param.dtype

        # We assume that all outputs are the last param!
        if idx == len(params) - 1:
            # Output tensor
            torch_dtype = _tvm_ffi_dtype_to_torch_dtype(dtype)
            output_torch = torch.zeros(shape, dtype=torch_dtype , device="cuda")
            output_tl = torch.zeros(shape, dtype=torch_dtype, device="cuda")
        else:
            # Input tensor
            if dtype == "float16":
                # Uniform distribution, N(0, 1), range (-1, 1) for float16.
                torch_tensor = torch.randn(shape, dtype=torch.float16, device="cuda")
            elif dtype == "float32":
                torch_tensor = torch.randn(shape, dtype=torch.float32, device="cuda")
            elif dtype == "uint8":
                torch_tensor = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")
            else:
                raise ValueError(f"Unsupported dtype: {dtype}, {type(dtype)}")
            inputs_in_torch_tensors.append(torch_tensor)

    return inputs_in_torch_tensors, output_torch, output_tl


def test_puzzle(puzzle_tl, puzzle_torch, hyper_params: dict, tl_hyper_params: dict = {}, print_log: bool=False):
    """Test a puzzle solution with given hyper parameters."""

    tl_kernel: JITKernel = puzzle_tl(**hyper_params, **tl_hyper_params)

    inputs_in_torch_tensors, output_torch, output_tl = _torch_tensor_materialize(tl_kernel.params)

    # As the kernel may modify the input tensors, we make a copy of them.
    inputs_copy = [i.clone() for i in inputs_in_torch_tensors]

    puzzle_torch(*inputs_copy, output_torch, **hyper_params)
    tl_kernel(*inputs_copy, output_tl)

    match = torch.allclose(output_torch, output_tl, rtol=1e-1)
    match_emoji = "✅" if match else "❌"
    print(match_emoji, "Results match:", match)

    if not match or print_log:
        print("Hyper parameters: ", hyper_params)
        print("Inputs: ", inputs_in_torch_tensors)
        print("Yours:", output_tl.dtype, output_tl.shape, "\n", output_tl)
        print("Spec:", output_torch.dtype, output_torch.shape, "\n", output_torch)
        print("Diff (True: correct, False: incorrect):", "\n", torch.isclose(output_torch, output_tl))


def bench_puzzle(puzzle_tl, puzzle_torch, hyper_params: dict, tl_hyper_params: dict = {}, bench_name: str = "Tilelang", bench_torch: bool = False):
    """Benchmark a puzzle solution with given hyper parameters."""

    warmups = 10
    repeats = 100

    tl_kernel: JITKernel = puzzle_tl(**hyper_params, **tl_hyper_params)

    inputs_in_torch_tensors, output_torch, output_tl = _torch_tensor_materialize(tl_kernel.params)

    if bench_torch:
        for _ in range(warmups):
            puzzle_torch(*inputs_in_torch_tensors, output_torch, **hyper_params)

        torch_start = torch.cuda.Event(enable_timing=True)
        torch_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        torch_start.record()
        for _ in range(repeats):
            puzzle_torch(*inputs_in_torch_tensors, output_torch, **hyper_params)
        torch_end.record()
        torch.cuda.synchronize()
        torch_time = torch_start.elapsed_time(torch_end) / repeats
        print(f"Torch time: {torch_time:.3f} ms")

    for _ in range(warmups):
        tl_kernel(*inputs_in_torch_tensors, output_tl)

    tl_start = torch.cuda.Event(enable_timing=True)
    tl_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    tl_start.record()
    for _ in range(repeats):
        tl_kernel(*inputs_in_torch_tensors, output_tl)
    tl_end.record()
    torch.cuda.synchronize()
    tl_time = tl_start.elapsed_time(tl_end) / repeats
    print(f"{bench_name} time: {tl_time:.3f} ms")

