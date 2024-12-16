import json
import numpy as np
import itertools

# Scoring functions
def weighted_sum(metrics, weights):
    return sum(metrics[metric] * weights[metric] for metric in weights)

def product_score(metrics):
    return np.prod(list(metrics.values()))

def average_score(metrics):
    return sum(metrics.values()) / len(metrics)

def weighted_product(metrics, weights):
    score = 1
    for metric in weights:
        score *= metrics[metric] ** weights[metric]
    return score

# Normalize PMI and Conditional Probability globally
def normalize_metrics(all_tuples_metrics):
    pmi_values = [metrics["PMI"] for metrics in all_tuples_metrics.values()]
    cp_values = [metrics["Conditional Probability"] for metrics in all_tuples_metrics.values()]

    min_pmi, max_pmi = min(pmi_values), max(pmi_values)
    min_cp, max_cp = min(cp_values), max(cp_values)

    for metrics in all_tuples_metrics.values():
        # Normalize PMI
        metrics["PMI_normalized"] = (metrics["PMI"] - min_pmi) / (max_pmi - min_pmi) if max_pmi != min_pmi else 1.0
        # Normalize Conditional Probability
        metrics["Conditional Probability_normalized"] = (metrics["Conditional Probability"] - min_cp) / (max_cp - min_cp) if max_cp != min_cp else 1.0

# Load the input JSON file
input_file = "../Output/word_metrics_nested.json"
with open(input_file, "r") as json_file:
    metrics_data = json.load(json_file)

# Flatten the dictionary to extract all tuples and their metrics
all_tuples_metrics = {}
for w1, w2_metrics in metrics_data.items():
    for w2, metrics in w2_metrics.items():
        all_tuples_metrics[(w1, w2)] = metrics

# Normalize PMI and Conditional Probability
normalize_metrics(all_tuples_metrics)


def generate_weight_variations(metrics, resolution=0.1):
    """
    Generate all possible weight combinations for the given metrics such that the sum of weights is 1.
    
    :param metrics: List of metric names.
    :param resolution: Step size for the weights (e.g., 0.1 for 10% increments).
    :return: List of dictionaries with weight variations.
    """
    # Generate all possible combinations of weights
    steps = int(1 / resolution) + 1
    possible_values = [i * resolution for i in range(steps)]  # E.g., [0.0, 0.1, 0.2, ..., 1.0]
    all_combinations = itertools.product(possible_values, repeat=len(metrics))
    
    # Filter combinations where the sum of weights is 1
    valid_combinations = [combo for combo in all_combinations if abs(sum(combo) - 1) < 1e-6]
    
    # Convert each combination into a dictionary
    weight_variations = [
        {metric: weight for metric, weight in zip(metrics, combo)}
        for combo in valid_combinations
    ]
    
    return weight_variations

# Define the metrics
metrics = ["PMI_normalized", "Conditional Probability_normalized", "Cosine Similarity", "Jaccard Similarity", "Dice Coefficient"]

# Generate weight variations with a resolution of 0.1
weight_variations = generate_weight_variations(metrics, resolution=0.2)

# Add calculated metrics to the input dictionary
for w1, w2_metrics in metrics_data.items():
    for w2, metrics in w2_metrics.items():
        # Prepare metrics for scoring
        metrics_for_scoring = {
            "PMI_normalized": metrics["PMI_normalized"],
            "Conditional Probability_normalized": metrics["Conditional Probability_normalized"],
            "Cosine Similarity": metrics["Cosine Similarity"],
            "Jaccard Similarity": metrics["Jaccard Similarity"],
            "Dice Coefficient": metrics["Dice Coefficient"]
        }

        # Calculate and add scores
        for i, weights in enumerate(weight_variations):
            metrics[f"Weighted Sum{i}"] = weighted_sum(metrics_for_scoring, weights)
            metrics[f"Weighted Product{i}"] = weighted_product(metrics_for_scoring, weights)
        metrics["Average"] = average_score(metrics_for_scoring)
        metrics["Product"] = product_score(metrics_for_scoring)

# Save the updated JSON with calculated metrics
output_file = "../Output/updated_metrics.json"
with open(output_file, "w") as json_file:
    json.dump(metrics_data, json_file, indent=4)

print(f"Updated metrics saved to {output_file}")
