# TileLang Puzzles Documentation

Welcome to the TileLang Puzzles documentation! This directory contains comprehensive learning materials for each puzzle, available in both Chinese and English.

## Directory Structure

```
docs/
├── zh/                    # 中文文档
│   ├── 1.copy/
│   ├── 2.vector-add/
│   ├── 3.outer-vec-add/
│   ├── 4.backward-op/
│   ├── 5.reduce-sum/
│   ├── 6.softmax/
│   ├── 7.scalar-flash-attn/
│   ├── 8.matrix/
│   ├── 9.conv/
│   └── 10.dequant-mm/
│
└── en/                    # English Documentation
    ├── 1.copy/
    ├── 2.vector-add/
    ├── 3.outer-vec-add/
    ├── 4.backward-op/
    ├── 5.reduce-sum/
    ├── 6.softmax/
    ├── 7.scalar-flash-attn/
    ├── 8.matrix/
    ├── 9.conv/
    └── 10.dequant-mm/
```

Each puzzle folder contains:
- **`1.{name}.md`** - Concept explanation and problem definition
- **`2.implementation-guide.md`** - Step-by-step implementation guide

## Puzzle Overview

| # | Puzzle | Key Concepts | 中文 | English |
|---|--------|--------------|------|---------|
| 01 | Copy | Memory copy, parallel threads, thread blocks | [中文](zh/1.copy/1.copy.md) | [English](en/1.copy/1.copy.md) |
| 02 | Vector Add | Element-wise operations, SIMD | [中文](zh/2.vector-add/1.vector-add.md) | [English](en/2.vector-add/1.vector-add.md) |
| 03 | Outer Vector Add | Broadcasting, memory access patterns | [中文](zh/3.outer-vec-add/1.outer-vec-add.md) | [English](en/3.outer-vec-add/1.outer-vec-add.md) |
| 04 | Backward Op | Gradient computation, backpropagation | [中文](zh/4.backward-op/1.backward-op.md) | [English](en/4.backward-op/1.backward-op.md) |
| 05 | Reduce Sum | Parallel reduction, atomic operations | [中文](zh/5.reduce-sum/1.reduce-sum.md) | [English](en/5.reduce-sum/1.reduce-sum.md) |
| 06 | Softmax | Numerical stability, online softmax | [中文](zh/6.softmax/1.softmax.md) | [English](en/6.softmax/1.softmax.md) |
| 07 | Scalar Flash Attention | Attention mechanism, memory optimization | [中文](zh/7.scalar-flash-attn/1.scalar-flash-attn.md) | [English](en/7.scalar-flash-attn/1.scalar-flash-attn.md) |
| 08 | Matrix | GEMM, tiling strategies, shared memory | [中文](zh/8.matrix/1.matrix.md) | [English](en/8.matrix/1.matrix.md) |
| 09 | Conv | Convolution, im2col, winograd | [中文](zh/9.conv/1.conv.md) | [English](en/9.conv/1.conv.md) |
| 10 | Dequant MM | Quantization, packed INT4 dequantized GEMM | [中文](zh/10.dequant-mm/1.dequant-mm.md) | [English](en/10.dequant-mm/1.dequant-mm.md) |

## Learning Path

We recommend completing the puzzles in order, as each builds upon concepts from previous ones:

```
Easy:     01 → 02 → 03 → 04 → 05
                  ↓
Medium:   06 → 07 → 08 → 09
                        ↓
Hard:                   10
```

### Difficulty Guide

- **Easy (01-05)**: Basic GPU programming concepts, memory operations, simple parallelism
- **Medium (06-09)**: Algorithm optimization, numerical stability, attention, matrix multiplication
- **Hard (10)**: Quantization techniques, real-world deployment scenarios

## Quick Start

1. Choose your preferred language: [中文](zh/) or [English](en/)
2. Start with Puzzle 01: Copy
3. Read the concept explanation first
4. Follow the implementation guide
5. Compare with reference implementation in `puzzles/ans/`

## Related Resources

- [TileLang GitHub](https://github.com/tile-ai/tilelang) - Official TileLang repository
- [Main README](../README.md) - Project setup and installation

## Contributing

Found an error or want to improve the documentation? Contributions are welcome!

- Report issues or suggest improvements via GitHub Issues
- Submit pull requests for documentation enhancements
