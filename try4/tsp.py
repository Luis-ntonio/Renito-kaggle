import heapq
import json
from gemma import *

model_path = "../model/gemma"
scorer = PerplexityCalculator(model_path=model_path, load_in_8bit=False)

def dictionary_to_graph(dictionary, metric="PMI"):
    """
    Converts a nested dictionary into an adjacency list representation of a graph.

    :param dictionary: A nested dictionary where keys are nodes and values are dictionaries of neighbors with weights.
    :return: An adjacency list as a dictionary of node -> list of (neighbor, weight).
    """
    graph = {}
    for node, edges in dictionary.items():
        graph[node] = [(neighbor, weight[metric]) for neighbor, weight in edges.items()]
    return graph


def tsp_dp(graph, start):
    """
    Solves the Traveling Salesman Problem (TSP) using Dynamic Programming (Held-Karp Algorithm).
    
    :param graph: An adjacency list as a dictionary of node -> list of (neighbor, weight).
    :param start: The starting node.
    :return: The optimal path and its cost.
    """
    # Step 1: Initialize
    nodes = list(graph.keys())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices
    start_index = node_index[start]

    # Convert graph to a distance matrix for simplicity
    dist = [[float('inf')] * n for _ in range(n)]
    for u, neighbors in graph.items():
        for v, weight in neighbors:
            dist[node_index[u]][node_index[v]] = weight

    # DP table: dp[mask][i] represents the minimum cost to visit all nodes in `mask` ending at `i`
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1 << start_index][start_index] = 0  # Starting point

    # Step 2: Fill DP table
    for mask in range(1 << n):  # Iterate over all subsets of nodes
        for u in range(n):  # Iterate over all ending nodes
            if not (mask & (1 << u)):  # If `u` is not in the subset `mask`, skip
                continue

            for v in range(n):  # Try to extend to node `v`
                if mask & (1 << v):  # If `v` is already visited, skip
                    continue

                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    # Step 3: Find the optimal path
    min_cost = float('inf')
    end_index = -1

    # Complete the tour back to the starting point
    for u in range(n):
        if u == start_index:
            continue
        cost = dp[(1 << n) - 1][u] + dist[u][start_index]
        if cost < min_cost:
            min_cost = cost
            end_index = u

    # Step 4: Reconstruct the path
    mask = (1 << n) - 1  # All nodes visited
    current = end_index
    path = [nodes[end_index]]

    while mask != (1 << start_index):
        for prev in range(n):
            if mask & (1 << prev) and dp[mask][current] == dp[mask ^ (1 << current)][prev] + dist[prev][current]:
                path.append(nodes[prev])
                mask ^= (1 << current)
                current = prev
                break

    path.reverse()  # Reverse the reconstructed path to start from the initial node

    return path, min_cost





with open("../Output/updated_metrics.json", "r") as json_file:
    metrics_data = json.load(json_file)

metrics_keys = list(list(metrics_data.values())[0].values())[0].keys()

# Solve the TSP using A*
best_cost = float('inf')
for metric in tqdm(metrics_keys):
    graph = dictionary_to_graph(metrics_data, metric=metric)
    for start_node in graph:
        path, cost = tsp_dp(graph, start_node)
        cost = scorer.get_perplexity(" ".join(path))
        if cost < best_cost:
            best_path = path
            best_cost = cost

print("Best path:", best_path)
print("Cost:", best_cost)