# icebug-benchmark-pagerank

Directed PageRank shootout between [icebug](https://github.com/Ladybug-Memory/icebug)
(a NetworKit fork backed by read-only CSR memory, `GraphR`), upstream
[networkit](https://github.com/networkit/networkit) (mutable weighted `GraphW`),
and [networkx](https://networkx.org/) (`DiGraph`, scipy-backed pagerank).
All engines run the same algorithm (`damp`/`alpha=0.85, tol=1e-6`); icebug and
networkit produce bit-identical scores. What differs is graph construction:

- **icebug**: zero-copy `Graph.fromCSR` from Arrow arrays (+ a transposed CSR for
  in-edge access, built with scipy) — no Python-level edge loop.
- **networkit**: edge-by-edge `addEdge` loop over the same CSR arrays.
- **networkx**: `DiGraph.add_edges_from` over CSR-derived edge lists.

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
cd networkx && uv sync && uv run bench.py [dataset-dir]     # default: wiki-Talk-csr
```

Examples:

```bash
cd icebug && uv run bench.py cit-Patents-csr
cd networkit && uv run bench.py cit-Patents-csr
cd networkx && uv run bench.py cit-Patents-csr
```

Peak memory and wall time are measured with `/usr/bin/time -v`, run from each
bench dir, e.g. `/usr/bin/time -v .venv/bin/python bench.py cit-Patents-csr`.

Threading follows `OMP_NUM_THREADS` (unset = all cores).

## Results

icebug and networkit produce identical top-10s on both graphs. networkx
reorders them (same nodes, different values) because it redistributes
dangling-node rank every iteration while NetworKit-family PageRank drops it
by default (`NoSinkHandling`).

Load = parquet reads, Build = graph construction (incl. transpose),
Runtime = PageRank only, Wall / Max RSS from `/usr/bin/time -v`.

### wiki-Talk (2,394,385 nodes / 5,021,410 edges, directed)

| | icebug (`GraphR`) | networkit (`GraphW`) | networkx (`DiGraph`) |
|---|---|---|---|
| Load | 0.08s | 0.08s | 0.08s |
| Build | **0.09s** | 4.08s | 7.88s |
| PageRank (runtime) | **0.15s** | 0.36s | 4.61s |
| Wall time | **0.92s** | 5.54s | 17.02s |
| Max RSS | **475 MiB** (486,360 kB) | 934 MiB (956,864 kB) | 3.40 GiB (3,564,404 kB) |
| Top node | 33: 2.5574e-04 | identical | 1764 (dangling handling differs) |

### cit-Patents (3,774,768 nodes / 16,518,947 edges, directed)

| | icebug (`GraphR`) | networkit (`GraphW`) | networkx (`DiGraph`) |
|---|---|---|---|
| Load | 0.16s | 0.16s | 0.17s |
| Build | **0.37s** | 12.27s | 27.31s |
| PageRank (runtime) | **0.62s** | 0.97s | 16.96s |
| Wall time | **1.89s** | 16.10s | 56.05s |
| Max RSS | **949 MiB** (971,588 kB) | 1.89 GiB (1,976,708 kB) | 8.74 GiB (9,160,908 kB) |
| Top node | 2031237: 8.7165e-05 | identical | 2475029 (dangling handling differs) |

Takeaways: parquet load is identical (~0.1–0.2s) since all three share the
same reader; construction dominates wall time for networkit/networkx while
icebug's zero-copy CSR handoff stays under half a second. `GraphR`'s
read-only CSR peaks at roughly half the RSS of mutable `GraphW`, and an
order of magnitude below networkx's Python-dict graph (8.74 GiB on
cit-Patents — fits, but barely, on a 12 GiB box).

## Notes / gotchas found while benchmarking

- `GraphR` stores out-edges only unless the transposed CSR is also passed to
  `fromCSR` (`in_indices`/`in_indptr`, documented as "only needed for directed
  graphs"). Directed PageRank traverses in-neighbors, so omitting them
  segfaults inside `GraphR::inNeighbors` (even on a 4-node graph) instead of
  raising. `icebug/bench.py` builds the true transpose with scipy for this reason.
- Upstream's constructor is `Graph(n, weighted, directed)` — passing a
  `directed` bool positionally as 2nd arg silently builds a weighted
  *undirected* graph. `networkit/bench.py` therefore uses
  `Graph(n_nodes, True, directed)` with keywords-by-position to get a directed
  `GraphW`. Always assert `isDirected()` / `isWeighted()` when it matters.
- PageRank scores were cross-validated engine-vs-engine (identical top-10s)
  and the transpose against brute-force neighbor scans before trusting any
  timing.
