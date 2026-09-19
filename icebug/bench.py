#!/usr/bin/env python3
import heapq
import sys
import time
from pathlib import Path

import icebug as ib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

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

def transpose_csr(indptr, indices):
    indptr_np = indptr.to_numpy()
    indices_np = indices.to_numpy()

    n = len(indptr_np) - 1
    m = len(indices_np)

    data = np.ones(m, dtype=np.int8)
    A = csr_matrix((data, indices_np, indptr_np), shape=(n, n))
    del indptr_np, indices_np, data

    # Pure transpose: in-edges only, no symmetrization, so the graph stays
    # directed and PageRank remains comparable with the networkit bench.
    T = A.transpose().tocsr()
    del A

    return T.indptr, T.indices


# GraphR stores out-edges only unless the transposed CSR is also given.
# Directed PageRank traverses in-neighbors, so build the true transpose
# here and keep the original CSR as out-edges; without in-edge storage
# GraphR::inNeighbors segfaults.
t = time.time()
t_indptr, t_indices = transpose_csr(indptr_arrow, indices_arrow)
in_indices = pa.array(t_indices, type=pa.uint64())
in_indptr = pa.array(t_indptr, type=pa.uint64())
print(f"Transpose built in {time.time()-t:.2f}s")

graph = ib.graph.Graph.fromCSR(
    n_nodes, directed, indices_arrow, indptr_arrow, in_indices, in_indptr
)
print(f"Created graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

start = time.time()
pr = ib.centrality.PageRank(graph, damp=0.85, tol=1e-6)
pr.run()
print(f"PageRank done in {time.time()-start:.5f}s, {len(pr.scores())} scores")
top_10 = heapq.nlargest(10, enumerate(pr.scores()), key=lambda x: x[1])
print("\nTop 10 nodes and scores:")
for i, (node, score) in enumerate(top_10):
    print(f"{i+1}. Node {node}: {score}")
