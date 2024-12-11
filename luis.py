"""Evaluation metric for Santa 2024."""

import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import logging
import numpy as np
import pandas as pd
import transformers
import torch
import pandas as pd
from gemma import *

import torch.nn.functional as F
import math

from tqdm import tqdm

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#save log to file
logging.basicConfig(filename='search.log', level=logging.INFO)

#start log
logging.info("Starting search")

model_path = "./model/gemma"
scorer = PerplexityCalculator(model_path=model_path, load_in_8bit=False)

data_input = pd.read_csv("./Input/sample_submission.csv")
Iterative_sentences = data_input['text'].tolist()[0:1]
print(Iterative_sentences)
#Iterative_sentences = [data_input['text'].tolist()[0]]

def get_dictionaries(tokens, encoded):
    dictionary = {}
    init_tokens = {}
    words = []
    current_word = ""
    idx = 0
    corr_enc = []
    for i, token in enumerate(tokens):
        if token.startswith('▁') or idx==0:
            idx += 1

            if current_word:
                words.append(current_word)
                dictionary[current_word] = corr_enc[len(words) - 1]
            # Remove the '▁' and start a new word
            if idx != 1:
                current_word = token[1:]
            else: 
                current_word = token[0:]
            corr_enc.append([encoded[i]])
            init_tokens[encoded[i]] = current_word
        else:
            # Continue the existing word
            corr_enc[len(words)].append(encoded[i])
            current_word += token

    # After the loop ends, add the last word if it exists
    if current_word:
        words.append(current_word)
        dictionary[current_word] = corr_enc[len(words) - 1]
    return dictionary, init_tokens, words

def get_tuples(dictionary):
    available_words = list(dictionary.keys())

    ppx_per_tuple = {}

    for word in available_words:
        remaining_words = [w for w in available_words if w != word]
        ppx_per_tuple[word] = {}
        for w in remaining_words:
            tuple_score = 0
            prompts = [ ["These are christmas words",1]] # A christmas sentence will appear after the dot.
            for prompt in prompts:
                tuple_score += scorer.get_perplexity(f"{prompt[0]} {word} {w}") * prompt[1]
            ppx_per_tuple[word][w] = tuple_score
    #sort the dictionary
    ppx_per_tuple = {k: v for k, v in sorted(ppx_per_tuple.items(), key=lambda item: item[0])}

    #sort sub dictionary for values
    for key in ppx_per_tuple:
        ppx_per_tuple[key] = {k: v for k, v in sorted(ppx_per_tuple[key].items(), key=lambda item: item[1], reverse=False)}
    return ppx_per_tuple

def get_triplets(dictionary):
    available_words = list(dictionary.keys())

    ppx_per_triplet = {}

    for word in available_words:
        remaining_words = [w for w in available_words if w != word]
        ppx_per_triplet[word] = {}
        for w1 in remaining_words:
            remaining_words_2 = [w for w in remaining_words if w != w1]
            ppx_per_triplet[word][w1] = {}
            for w2 in remaining_words_2:
                triplet_score = scorer.get_perplexity(
                    f"A christmas sentence will appear after the dot. {word} {w1} {w2}"
                )
                ppx_per_triplet[word][w1][w2] = triplet_score

    # Sort the outer dictionary
    ppx_per_triplet = {k: v for k, v in sorted(ppx_per_triplet.items(), key=lambda item: item[0])}

    # Sort sub-dictionaries
    for key in ppx_per_triplet:
        ppx_per_triplet[key] = {k: v for k, v in sorted(ppx_per_triplet[key].items(), key=lambda item: item[0])}
        for sub_key in ppx_per_triplet[key]:
            ppx_per_triplet[key][sub_key] = {
                k: v for k, v in sorted(ppx_per_triplet[key][sub_key].items(), key=lambda item: item[1])
            }

    return ppx_per_triplet

def find_lowest_value_sentence(data):
    returns = []
    all_words = set(data.keys())
    
    for word in all_words:
        sentence = []
        current_word = word
        remaining_words = set([w for w in all_words if w != word])
        sentence.append(current_word)

        while len(remaining_words) > 0:
            for k, v in data[current_word].items():
                if k not in sentence:
                    sentence.append(k)
                    current_word = k
                    remaining_words.remove(k)
                    break
        returns.append(" ".join(sentence))
    return returns

def find_lowest_value_sentence_triplets(data):
    returns = []
    all_words = set(data.keys())
    logging.info("Starting lowest value sentence with words {}".format(all_words))

    for word in tqdm(all_words):
        sentence = []
        current_word = word
        remaining_words = set([w for w in all_words if w != word])
        sentence.append(current_word)

        while len(remaining_words) > 0:
            # Handle the case where only one word is left
            if len(remaining_words) == 1:
                last_word = remaining_words.pop()
                sentence.append(last_word)
                break

            found = False
            if len(sentence) < 2:
                for next_word, sub_dict in data[current_word].items():
                    if next_word not in sentence:
                        for third_word, _ in sub_dict.items():
                            if third_word not in sentence:
                                if len(sentence) < 2:
                                    sentence.extend([next_word, third_word])
                                else:
                                    sentence.append(third_word)
                                current_word = next_word
                                remaining_words -= {next_word, third_word}
                                found = True
                                break
                    if found:
                        break
            else:
                for next_word, _ in data[current_word][third_word].items():
                    if next_word not in sentence:
                        sentence.append(next_word)
                        current_word = third_word
                        third_word = next_word
                        remaining_words -= {next_word}
                        break

        returns.append(" ".join(sentence))
    return returns

def dictionary_to_graph(dictionary):
    """
    Converts a nested dictionary into an adjacency list representation of a graph.

    :param dictionary: A nested dictionary where keys are nodes and values are dictionaries of neighbors with weights.
    :return: An adjacency list as a dictionary of node -> list of (neighbor, weight).
    """
    graph = {}
    for node, edges in dictionary.items():
        graph[node] = [(neighbor, weight) for neighbor, weight in edges.items()]
    return graph

import heapq

def tsp_a_star(graph, start):
    """
    Solves the Traveling Salesman Problem (TSP) using A*,
    ensuring each node is visited only once.
    
    :param graph: An adjacency list as a dictionary of node -> list of (neighbor, weight).
    :param start: The starting node.
    :return: The shortest path that visits all nodes and its cost.
    """
    n = len(graph)  # Number of nodes
    nodes = list(graph.keys())
    node_index = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices for bitmasking
    
    # Priority queue for A*
    pq = []
    initial_state = (0, start, 1 << node_index[start], [start])  # (cost, current_node, visited_mask, path)
    heapq.heappush(pq, initial_state)
    
    # Store the minimum cost for a given node and visited state
    min_cost = {}
    
    while pq:
        cost, current, visited, path = heapq.heappop(pq)
        
        # If all nodes are visited, return the path and cost
        if visited == (1 << n) - 1:  # All nodes visited
            return path, cost
        
        # Skip states that have already been processed with a lower cost
        if (current, visited) in min_cost and min_cost[(current, visited)] <= cost:
            continue
        min_cost[(current, visited)] = cost
        
        # Explore neighbors
        for neighbor, weight in graph[current]:
            neighbor_index = node_index[neighbor]
            
            # If the neighbor is already visited, skip
            if visited & (1 << neighbor_index):
                continue
            
            # Update the visited state and push the new state to the queue
            next_visited = visited | (1 << neighbor_index)
            next_state = (cost + weight, neighbor, next_visited, path + [neighbor])
            heapq.heappush(pq, next_state)
    
    # If no solution is found (should not happen if the graph is connected)
    return None, float('inf')

output = []
import json

for sen in Iterative_sentences:
    encoded = scorer.tokenizer.encode(sen, add_special_tokens=False)
    tokens = scorer.tokenizer.convert_ids_to_tokens(encoded)
    logging.info("Finishing encoding")
    dictionary, init_tokens, words = get_dictionaries(tokens, encoded)
    logging.info("Finishing dictionaries")
    ppx_per_tuple = get_tuples(dictionary)
    with open('dictionary.json', 'w') as f:
        json.dump(ppx_per_tuple, f, indent=4)
    ppx_graph = dictionary_to_graph(ppx_per_tuple)
    logging.info("Finishing triplets")
    #sentence = find_lowest_value_sentence(ppx_per_tuple)
    sentence = []
    for word in words:

        sent, cost = tsp_a_star(ppx_graph, word)
        sent = " ".join(sent)
        sentence.append(sent)
    print(sentence)
    logging.info("Finishing lowest value sentence")

    solution_final = ""
    score = float("inf")
    for sol in tqdm(sentence):
        solutions = pd.DataFrame(
            {'id': [0],
            'text': sol})

        perplexities = scorer.get_perplexity(solutions["text"].tolist(), debug=False)
        if perplexities[0] < score:
            score = perplexities[0]
            solution_final = sol

    output.append(solution_final)
    """senten = " ".join(sorted(sen.split(" ")))
    print(senten)
    output.append(senten)"""


solutions = pd.DataFrame(
            {'id': [x for x in range(len(output))],
            'text': output})

perplexities = scorer.get_perplexity(solutions["text"].tolist(), debug=False)
logging.info("Perplexities: {} {}".format(perplexities, output))
