from tqdm import tqdm
import json
from itertools import permutations
import numpy as np
from math import log
from gemma import *

model_path = "../model/gemma"
scorer = PerplexityCalculator(model_path=model_path, load_in_8bit=False)
scorer.model.eval()


# Sentence and tokenize
sentence = "scrooge mistletoe ornament family advent fireplace chimney elf reindeer gingerbread"
words = sentence.split()

word_embeddings = {}
for word in words:
    # Tokenize the word
    tokens = scorer.tokenizer.encode(word, return_tensors="pt")
    
    # Enable hidden states
    scorer.model.config.output_hidden_states = True
    
    # Get embeddings from the model
    with torch.no_grad():
        outputs = scorer.model(input_ids=tokens)
        hidden_states = outputs.hidden_states  # Get all hidden states
    
    # Use the last hidden layer as the word embedding
    embedding = hidden_states[-1].squeeze(0).mean(dim=0)  # Mean pool across tokens for single word
    word_embeddings[word] = embedding.cpu().numpy()  # Store as numpy array for further use


word_probabilities = {}

for word in words:
    perplexity = scorer.get_perplexity(word)
    perplexity = perplexity**0.25
    probability = 2 ** (-perplexity)
    word_probabilities[word] = probability

joint_probabilities = {}

for w1, w2 in permutations(words, 2):
    pair = f"{w1} {w2}"  # Combine words into a pair
    perplexity = scorer.get_perplexity(pair)
    perplexity = perplexity**0.25
    joint_probability = 2 ** (-perplexity)
    joint_probabilities[(w1, w2)] = joint_probability


# Initialize dictionary to store results
results = {}

# Metrics functions
def calculate_pmi(joint_prob, prob1, prob2):
    return log(joint_prob / (prob1 * prob2), 2) if joint_prob > 0 else 0

def calculate_conditional_probability(joint_prob, prob1):
    return joint_prob / prob1 if prob1 > 0 else 0

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0

def dice_coefficient(set1, set2):
    intersection = len(set1 & set2)
    return float(2 * intersection / (len(set1) + len(set2))) if (len(set1) + len(set2)) > 0 else 0

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0

# Compute metrics for all tuples
for w1, w2 in tqdm(permutations(words, 2)):
    joint_prob = joint_probabilities[(w1, w2)]
    prob1 = word_probabilities[w1]
    prob2 = word_probabilities[w2]

    pmi = calculate_pmi(joint_prob, prob1, prob2)
    conditional_prob = calculate_conditional_probability(joint_prob, prob1)
    cosine_sim = cosine_similarity(word_embeddings[w1], word_embeddings[w2])

    # Example sets for Jaccard and Dice (replace with actual co-occurrence sets)
    set1, set2 = set(w1), set(w2)  # Placeholder sets

    jaccard = jaccard_similarity(set1, set2)
    dice = dice_coefficient(set1, set2)

    # Nested structure for results
    if w1 not in results:
        results[w1] = {}
    results[w1][w2] = {
        "PMI": pmi,
        "Conditional Probability": conditional_prob,
        "Cosine Similarity": cosine_sim,
        "Jaccard Similarity": jaccard,
        "Dice Coefficient": dice
    }

# Save results to JSON file
with open("../Output/word_metrics_nested.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to word_metrics_nested.json")
