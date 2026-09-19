# icebug-benchmark-pagerank

Directed PageRank shootout between [icebug](https://github.com/Ladybug-Memory/icebug)
(a NetworKit fork backed by read-only CSR memory, `GraphR`) and upstream
[networkit](https://github.com/networkit/networkit) (mutable weighted `GraphW`).
Both engines run the same algorithm (`damp=0.85, tol=1e-6`) and produce
bit-identical scores; what differs is graph construction:

- **icebug**: zero-copy `Graph.fromCSR` from Arrow arrays (+ a transposed CSR for
  in-edge access, built with numpy) — no Python-level edge loop.
- **networkit**: edge-by-edge `addEdge` loop over the same CSR arrays.

## Datasets

Real connected graphs from
[ladybugdb/ldbc-csr](https://huggingface.co/datasets/ladybugdb/ldbc-csr),
loaded directly from Parquet (columns `id` / `target` / `ptr`). Both are directed.

| Dataset | Nodes | Edges | Dir |
|---|---|---|---|
| `wiki-Talk-csr` | 2,394,385 | 5,021,410 | `wiki-Talk-csr/` |
| `cit-Patents-csr` | 3,774,768 | 16,518,947 | `cit-Patents-csr/` |

Download (4 files per dataset: `indices_*_rel.parquet`, `indptr_*_rel.parquet`,
`nodes_*.parquet`, `schema.cypher`):

```bash
mkdir -p wiki-Talk-csr cit-Patents-csr
for f in indices_wiki_Talk_rel.parquet indptr_wiki_Talk_rel.parquet nodes_wiki_Talk.parquet schema.cypher; do
  curl -s -L -o wiki-Talk-csr/$f "https://huggingface.co/datasets/ladybugdb/ldbc-csr/resolve/main/wiki-Talk-csr/$f"
done
for f in indices_cit_Patents_rel.parquet indptr_cit_Patents_rel.parquet nodes_cit_Patents.parquet schema.cypher; do
  curl -s -L -o cit-Patents-csr/$f "https://huggingface.co/datasets/ladybugdb/ldbc-csr/resolve/main/cit-Patents-csr/$f"
done
```

(This replaces the old `csr_graph.db` benchmark, which was bogus: mostly
disconnected nodes, run undirected.)

## Running benchmarks

Each side has its own venv so the engines stay isolated — the icebug venv must
resolve `import icebug` to the fork (which provides `fromCSR`), the networkit
venv to upstream (which does not have `fromCSR`):

```bash
cd icebug && uv sync && uv run bench.py [dataset-dir]       # default: wiki-Talk-csr
cd networkit && uv sync && uv run bench.py [dataset-dir]    # default: wiki-Talk-csr
```

Examples:

```bash
cd icebug && uv run bench.py cit-Patents-csr
cd networkit && uv run bench.py cit-Patents-csr
```

Threading follows `OMP_NUM_THREADS` (unset = all cores).

## Results

Top-10 nodes and scores are identical between engines on both graphs.

### wiki-Talk (2,394,385 nodes / 5,021,410 edges, directed)

| | icebug (`GraphR`) | networkit (`GraphW`) |
|---|---|---|
| Build / load | transpose 0.31s | edge-by-edge ~4s |
| PageRank | **0.15s** | 0.36s |
| Top node | 33: 2.5574e-04 | identical |

### cit-Patents (3,774,768 nodes / 16,518,947 edges, directed)

| | icebug (`GraphR`) | networkit (`GraphW`) |
|---|---|---|
| Build / load | transpose 1.87s | edge-by-edge ~12s |
| PageRank | **0.57s** | 1.00s |
| Max RSS | **1.12 GiB** (1,179,368 kB) | 1.89 GiB (1,976,648 kB) |
| Top node | 2031237: 8.7165e-05 | identical |

Max RSS measured with `/usr/bin/time -v`, run from each bench dir as
`/usr/bin/time -v .venv/bin/python bench.py cit-Patents-csr`
(icebug: 3.31s wall; networkit: 15.92s wall). `GraphR`'s read-only CSR
uses ~40% less peak memory than the mutable `GraphW` on this graph.

## Notes / gotchas found while benchmarking

- `GraphR` stores out-edges only unless the transposed CSR is also passed to
  `fromCSR` (`in_indices`/`in_indptr`, documented as "only needed for directed
  graphs"). Directed PageRank traverses in-neighbors, so omitting them
  segfaults inside `GraphR::inNeighbors` (even on a 4-node graph) instead of
  raising. `icebug/bench.py` builds the transpose with numpy for this reason.
- Upstream's constructor is `Graph(n, weighted, directed)` — passing a
  `directed` bool positionally as 2nd arg silently builds a weighted
  *undirected* graph. `networkit/bench.py` therefore uses
  `Graph(n_nodes, True, directed)` with keywords-by-position to get a directed
  `GraphW`. Always assert `isDirected()` / `isWeighted()` when it matters.
- PageRank scores were cross-validated engine-vs-engine (identical top-10s)
  and the transpose against brute-force neighbor scans before trusting any
  timing.
