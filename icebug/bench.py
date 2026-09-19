#!/usr/bin/env python3
import heapq
import sys
import time
from pathlib import Path

import icebug as nk
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / (sys.argv[1] if len(sys.argv) > 1 else "wiki-Talk-csr")

n_nodes = pq.read_table(next(DATA_DIR.glob("nodes_*.parquet")), columns=["id"]).num_rows
indices_arrow = pq.read_table(next(DATA_DIR.glob("indices_*.parquet")), columns=["target"])[
    "target"
].combine_chunks()
indptr_arrow = pq.read_table(next(DATA_DIR.glob("indptr_*.parquet")), columns=["ptr"])[
    "ptr"
].combine_chunks()
n_edges = len(indices_arrow)
directed = True  # both LDBC CSR graphs here are directed
assert len(indptr_arrow) == n_nodes + 1
print(f"Graph: {n_nodes} nodes, {n_edges} edges, directed={directed}")

print(f"CSR arrays: indices={len(indices_arrow)}, indptr={len(indptr_arrow)}")

# GraphR stores out-edges only unless the transposed CSR is also given.
# Directed PageRank traverses in-neighbors, so build the transpose
# (CSC) here; without it GraphR::inNeighbors segfaults.
t = time.time()
out_ptr = indptr_arrow.to_numpy().astype(np.int64)
out_idx = indices_arrow.to_numpy().astype(np.int64)
sources = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(out_ptr))
order = np.argsort(out_idx, kind="stable")
in_indices = pa.array(sources[order], type=pa.uint64())
in_indptr = pa.array(np.concatenate(([0], np.cumsum(np.bincount(out_idx, minlength=n_nodes)))), type=pa.uint64())
print(f"Transpose built in {time.time()-t:.2f}s")

graph = nk.graph.Graph.fromCSR(n_nodes, directed, indices_arrow, indptr_arrow, in_indices, in_indptr)
print(f"Created graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

start = time.time()
pr = nk.centrality.PageRank(graph, damp=0.85, tol=1e-6)
pr.run()
print(f"PageRank done in {time.time()-start:.5f}s, {len(pr.scores())} scores")
top_10 = heapq.nlargest(10, enumerate(pr.scores()), key=lambda x: x[1])
print("\nTop 10 nodes and scores:")
for i, (node, score) in enumerate(top_10):
    print(f"{i+1}. Node {node}: {score}")
