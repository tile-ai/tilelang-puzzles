<img src=./images/logo-row.svg />

<div align="center">

# TileLang Puzzles

</div>

TileLang Puzzles is a set of puzzles to help you learn [TileLang](https://github.com/tile-ai/tilelang), a domain-specific language for developing high-performance deep learning kernels. We will start from some trivial examples and smoothly progress to modern kernels such as GEMM and FlashAttention, aiming to provide a comprehensive understanding of the design principles of TileLang.

## Getting Started

#### Environment Configuration

The only thing you need to install is [TileLang](https://github.com/tile-ai/tilelang) and its dependency. To check your installation, run:

```bash
python -c "import tilelang; print(tilelang.__version__);"
```

You can also run our environment check script to confirm that TileLang and your GPU are configured correctly:

```bash
python3 scripts/check_tilelang_env.py
```

#### Run Puzzles

We provide 10 puzzles in the `puzzles/` directory, each as a standalone executable script. Reference implementations are available in `ans/` for comparison. To run a puzzle, use:

```bash
python3 puzzles/01-copy.py
python3 ans/01-copy.py
```

## Acknowledgements

This is project is inspired by the following projects: [Triton Puzzles](https://github.com/srush/Triton-Puzzles) & [Triton Puzzles Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite), [LeetGPU](https://leetgpu.com/).
