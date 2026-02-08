#!/usr/bin/env python3
import duckdb
import heapq
import networkit as nk
import pyarrow as pa
import time

conn = duckdb.connect("csr_graph.db")
metadata = conn.execute("SELECT * FROM csr_graph_metadata").pl()
n_nodes = int(metadata["n_nodes"][0])
n_edges = int(metadata["n_edges"][0])
directed = bool(metadata["directed"][0])
print(f"Graph: {n_nodes} nodes, {n_edges} edges, directed={directed}")

indices_arrow = (
    conn.execute("SELECT target::ubigint as target FROM csr_graph_indices_edges")
    .fetch_arrow_table()["target"]
    .combine_chunks()
)
indptr_arrow = (
    conn.execute("SELECT ptr::ubigint as ptr FROM csr_graph_indptr_edges")
    .fetch_arrow_table()["ptr"]
    .combine_chunks()
)


print(f"CSR arrays: indices={len(indices_arrow)}, indptr={len(indptr_arrow)}")

graph = nk.graph.Graph.fromCSR(n_nodes, directed, indices_arrow, indptr_arrow)
print(f"Created graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

start = time.time()
pr = nk.centrality.PageRank(graph, damp=0.85, tol=1e-6)
pr.run()
print(f"PageRank done in {time.time()-start:.5f}s, {len(pr.scores())} scores")
top_10 = heapq.nlargest(10, enumerate(pr.scores()), key=lambda x: x[1])
print("\nTop 10 nodes and scores:")
for i, (node, score) in enumerate(top_10):
    print(f"{i+1}. Node {node}: {score}")
