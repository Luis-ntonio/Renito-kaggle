from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

from gemma import *

model_path = "./model/gemma"
scorer = PerplexityCalculator(model_path=model_path, load_in_8bit=False)

# Ensure the model is in evaluation mode
scorer.model.eval()


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

def evaluate_probs(sentence, subvocab_ids, dictionary):
    with torch.no_grad():
        text_with_special = f"{scorer.tokenizer.bos_token}{sentence}{scorer.tokenizer.eos_token}"

        # Tokenize
        model_inputs = scorer.tokenizer(
            text_with_special,
            return_tensors='pt',
            add_special_tokens=False,
        )

        if 'token_type_ids' in model_inputs:
            model_inputs.pop('token_type_ids')

        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

        # Get model output
        outputs = scorer.model(**model_inputs, use_cache=False)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    # Get the logits for the last token in the sequence
    last_token_logits = logits[:, -1, :].squeeze(0)

    # Mask out logits not in the subvocabulary
    mask = torch.ones_like(last_token_logits) * float("-inf")
    mask[subvocab_ids] = 0
    masked_logits = last_token_logits + mask

    # Compute probabilities for the subvocabulary
    subvocab_probs = F.softmax(masked_logits[subvocab_ids], dim=0)

    # Sample or choose the next token from the subvocabulary
    subvocab_probs = subvocab_probs.tolist()
    returns = {}
    for predicted_index in range(len(subvocab_probs)):
        predicted_token_id = subvocab_ids[predicted_index]
        predicted_token = scorer.tokenizer.decode(predicted_token_id)
        #find position in word that appears the subword
        
        for word in dictionary:
            tmp = " "+ word
            position = tmp.find(predicted_token)
            if position != -1 and position < 2:
                if word not in returns:
                    returns[word] = subvocab_probs[predicted_index]
                else:
                    returns[word] += subvocab_probs[predicted_index]
    
    return returns

def get_probs(dictionary, enc):
    available_words = list(dictionary.keys())

    ppx_per_tuple = {}

    for word in available_words:
        tokens = dictionary[word]
        remaining_words = [w for w in available_words if w != word]
        ppx_per_tuple[word] = {}
        encodings = enc.copy()
        for t in tokens:
            encodings.remove(t)

        tuple_score = {}
        prompts = [ ["These are christmas words",1]] # A christmas sentence will appear after the dot.
        for prompt in prompts:
            tuple_score = evaluate_probs(f"{prompt[0]} {word}", encodings, dictionary) 
        ppx_per_tuple[word] = tuple_score
    #sort the dictionary
    ppx_per_tuple = {k: v for k, v in sorted(ppx_per_tuple.items(), key=lambda item: item[0])}

    #sort sub dictionary for values
    for key in ppx_per_tuple:
        ppx_per_tuple[key] = {k: v for k, v in sorted(ppx_per_tuple[key].items(), key=lambda item: item[1], reverse=False)}
    return ppx_per_tuple

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

def merge_dicts_with_product(dict1, dict2):

    """

    Merge two nested dictionaries by multiplying the values of matching keys and subkeys.



    Args:

    - dict1 (dict): First nested dictionary.

    - dict2 (dict): Second nested dictionary.



    Returns:

    - dict: A merged dictionary with products of values from matching keys and subkeys.

    """

    merged_dict = {}

    eps = 1e-10  # Small value to prevent log(0)

    for key, sub_dict1 in dict1.items():

        merged_dict[key] = {}

        sub_dict2 = dict2.get(key, {})

        for subkey, value1 in sub_dict1.items():

            value2 = sub_dict2.get(subkey, 1)  # Default to 1 if subkey doesn't exist in dict2

            merged_dict[key][subkey] = (value1)# * value2)
            if merged_dict[key][subkey] < 1:
                del merged_dict[key][subkey]
            else:
                merged_dict[key][subkey] = math.log((merged_dict[key][subkey])**0.5 + eps)
    # Calculate the median of the values in merged_dict[key]
    for key in merged_dict:
        values = list(merged_dict[key].values())
        mean = sum(values) / len(values)
        median = sorted(values)[len(values) // 2]
        mean += 1.25*(median - mean)  # Add 5% of the difference between mean and median
                    
        keys_to_delete = []
        for subkey in merged_dict[key]:
            if merged_dict[key][subkey] > mean:
                keys_to_delete.append(subkey)
        for subkey in keys_to_delete:
            del merged_dict[key][subkey]

    merged_dict = {k: v for k, v in sorted(merged_dict.items(), key=lambda item: item[0])}
    for key in merged_dict:
        merged_dict[key] = {k: v for k, v in sorted(merged_dict[key].items(), key=lambda item: item[1], reverse=False)}

    return merged_dict


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


# Define the subvocabulary

data_input = pd.read_csv("./Input/sample_submission.csv")
Iterative_sentences = data_input['text'].tolist()[0:1]





output = []
for sen in Iterative_sentences:
    encode = scorer.tokenizer.encode(sen, add_special_tokens=False)
    tokens = scorer.tokenizer.convert_ids_to_tokens(encode)

    dictionary, init_tokens, words = get_dictionaries(tokens, encode)
    #probs_per_tuple = get_probs(dictionary, encode)
    ppx_per_tuple = get_tuples(dictionary)

    merged_tuple = merge_dicts_with_product(ppx_per_tuple, ppx_per_tuple)
    import json
    with open('merged.json', 'w') as f:
        json.dump(merged_tuple, f, indent=4)

    ppx_graph = dictionary_to_graph(merged_tuple)
    sentence = []
    for word in words:

        sent, cost = tsp_a_star(ppx_graph, word)
        print(sent, cost)
        if sent is None:
            continue
        print(" ".join(sent))
        sent = " ".join(sent)
        sentence.append(sent)

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


solutions = pd.DataFrame(
            {'id': [x for x in range(len(output))],
            'text': output})

perplexities = scorer.get_perplexity(solutions["text"].tolist(), debug=False)
print(output)
print(perplexities)