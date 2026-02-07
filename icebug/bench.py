#!/usr/bin/env python3
import duckdb
import networkit as nk
import pyarrow as pa
import time

conn = duckdb.connect('csr_graph.db')
metadata = conn.execute('SELECT * FROM csr_graph_metadata').pl()
n_nodes = int(metadata['n_nodes'][0])
n_edges = int(metadata['n_edges'][0])
directed = bool(metadata['directed'][0])
print(f'Graph: {n_nodes} nodes, {n_edges} edges, directed={directed}')

indices_pl = conn.execute('SELECT target FROM csr_graph_indices_edges').pl()
indices_arrow = pa.array(indices_pl['target'].to_list(), type=pa.uint64())

indptr_pl = conn.execute('SELECT ptr FROM csr_graph_indptr_edges').pl()
indptr_arrow = pa.array(indptr_pl['ptr'].to_list(), type=pa.uint64())

print(f'CSR arrays: indices={len(indices_arrow)}, indptr={len(indptr_arrow)}')

nk.setNumberOfThreads(1)
graph = nk.graph.Graph.fromCSR(n_nodes, directed, indices_arrow, indptr_arrow)
print(f'Created graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges')

start = time.time()
pr = nk.centrality.PageRank(graph, damp=0.85, tol=1e-8)
pr.run()
print(f'PageRank done in {time.time()-start:.5f}s, {len(pr.scores())} scores')
