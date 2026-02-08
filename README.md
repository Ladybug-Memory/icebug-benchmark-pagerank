# icebug-benchmark

This repository compares the performance of [icebug](https://github.com/Ladybug-Memory/icebug) and [networkit](https://github.com/networkit/networkit), two graph processing libraries.

## Libraries Compared

### icebug
A high-performance graph processing library designed for efficient memory management and parallel computation.

### networkit
A toolkit for large-scale network analysis with optimized algorithms for graph processing and network metrics.

## Benchmark Scope

The benchmarks in this repository evaluate various aspects of graph processing, including:

- Algorithm execution time
- Memory usage
- Scalability with different graph sizes
- Parallelization efficiency

## Dataset

The benchmarks use the [wikidata-csr-200m](https://huggingface.co/datasets/ladybugdb/wikidata-csr-200m) dataset from HuggingFace, which contains the `csr_graph.db` file with large-scale graph data for benchmarking graph processing libraries.

## Running Benchmarks

Instructions for running benchmarks can be found in the individual benchmark files.

## icebug venv Setup

icebug is not yet available as a pip package. To set up the icebug virtual environment:

1. Clone the icebug repository:
   ```bash
   git clone https://github.com/Ladybug-Memory/icebug.git
   cd icebug
   ```

2. Install icebug in editable mode:
   ```bash
   uv pip install -e .
   ```

3. Return to the benchmark directory and run benchmarks using the icebug venv.

## Benchmark Results (single threaded)

| Metric | icebug | networkit |
|--------|--------|-----------|
| **PageRank Runtime** | 7.55s | 60.70s |
| **Total Wall Time** | 23.39s | 3m 3.72s |
| **Max Memory Usage** | 8.53 GB | 12.07 GB |

**Speedup**: icebug is **8x faster** for PageRank computation.

**Memory**: icebug uses **29% less memory** than networkit.

*Graph tested: 115,475,324 nodes, 20,000,000 edges (undirected)*

## Benchmark Setup (multi-threaded)

Run with OMP_NUM_THREADS=8. Higher number of threads and tight convergance such as tol=1e-8 are known
to be problematic.
