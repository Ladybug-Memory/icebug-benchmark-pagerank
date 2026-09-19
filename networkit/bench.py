#!/usr/bin/env python3
import heapq
import sys
import time
from pathlib import Path
from tqdm import tqdm

import networkit as nk
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

graph = nk.graph.Graph(n_nodes, True, directed)  # weighted + directed
for u in tqdm(range(n_nodes), desc="Building graph"):
    start = indptr_arrow[u].as_py()
    end = indptr_arrow[u + 1].as_py()
    for idx in range(start, end):
        v = indices_arrow[idx].as_py()
        graph.addEdge(u, v)
print(f"Created graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

start = time.time()
pr = nk.centrality.PageRank(graph, damp=0.85, tol=1e-6)
pr.run()
print(f"PageRank done in {time.time()-start:.5f}s, {len(pr.scores())} scores")
top_10 = heapq.nlargest(10, enumerate(pr.scores()), key=lambda x: x[1])
print("\nTop 10 nodes and scores:")
for i, (node, score) in enumerate(top_10):
    print(f"{i+1}. Node {node}: {score}")
