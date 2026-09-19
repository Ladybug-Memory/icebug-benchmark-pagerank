#!/usr/bin/env python3
import heapq
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / (sys.argv[1] if len(sys.argv) > 1 else "wiki-Talk-csr")

t = time.time()
n_nodes = pq.read_table(next(DATA_DIR.glob("nodes_*.parquet")), columns=["id"]).num_rows
indices_arrow = pq.read_table(next(DATA_DIR.glob("indices_*.parquet")), columns=["target"])[
    "target"
].combine_chunks()
indptr_arrow = pq.read_table(next(DATA_DIR.glob("indptr_*.parquet")), columns=["ptr"])[
    "ptr"
].combine_chunks()
print(f"Load done in {time.time()-t:.2f}s")
n_edges = len(indices_arrow)
directed = True  # both LDBC CSR graphs here are directed
assert len(indptr_arrow) == n_nodes + 1
print(f"Graph: {n_nodes} nodes, {n_edges} edges, directed={directed}")

print(f"CSR arrays: indices={len(indices_arrow)}, indptr={len(indptr_arrow)}")

t = time.time()
indptr_np = indptr_arrow.to_numpy().astype(np.int64)
indices_np = indices_arrow.to_numpy().astype(np.int64)
sources = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr_np)).tolist()
targets = indices_np.tolist()
del indptr_np, indices_np
graph = nx.DiGraph()
graph.add_nodes_from(range(n_nodes))
graph.add_edges_from(zip(sources, targets))
del sources, targets
print(f"Build done in {time.time()-t:.2f}s")
print(f"Created graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

start = time.time()
scores = nx.pagerank(graph, alpha=0.85, tol=1e-6)
print(f"PageRank done in {time.time()-start:.5f}s, {len(scores)} scores")
top_10 = heapq.nlargest(10, scores.items(), key=lambda x: x[1])
print("\nTop 10 nodes and scores:")
for i, (node, score) in enumerate(top_10):
    print(f"{i+1}. Node {node}: {score}")
