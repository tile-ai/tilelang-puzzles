# TileLang Puzzles

TileLang Puzzles is a set of puzzles to help you learn [TileLang](https://github.com/tile-ai/tilelang), a domain-specific language for developing high-performance deep learning kernels. We will start from some trivial examples and smoothly progress to modern kernels such as GEMM and FlashAttention, aiming to provide a comprehensive understanding of the design principles of TileLang.

## Environment Configuration

The only thing you need to install is [TileLang](https://github.com/tile-ai/tilelang) and its dependency. To check your installation, run:

```python
python -c "import tilelang; print(tilelang.__version__);"
```

## Acknowledgements

This is project is inspired by the following projects: [Triton Puzzles](https://github.com/srush/Triton-Puzzles), [LeetGPU](https://leetgpu.com/).