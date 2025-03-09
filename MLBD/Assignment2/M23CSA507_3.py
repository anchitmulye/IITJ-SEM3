import math
import os
from itertools import combinations


def s_curve(x, a=1, b=1):
    return 1 / (1 + math.exp(-a * (x - b)))


def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()


def character_k_grams(text, k):
    return set(text[i:i + k] for i in range(len(text) - k + 1))


def word_k_grams(text, k):
    words = text.split()
    return set(tuple(words[i:i + k]) for i in range(len(words) - k + 1))


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


# Load documents
documents = {}
for i in range(1, 5):  # D1 to D4
    file_path = f"minhash/D{i}.txt"
    if os.path.exists(file_path):
        documents[f"D{i}"] = read_document(file_path)
    else:
        print(f"Warning: {file_path} not found.")

k_values = [(2, 'char'), (3, 'char'), (2, 'word')]
kgram_data = {}

for k, gram_type in k_values:
    for doc_name, text in documents.items():
        if gram_type == 'char':
            kgram_data[(doc_name, k, 'char')] = character_k_grams(text, k)
        else:
            kgram_data[(doc_name, k, 'word')] = word_k_grams(text, k)

kgram_counts = {}
for (doc, k, gram_type), kgram_set in kgram_data.items():
    count = len(kgram_set)
    kgram_counts[(doc, k, gram_type)] = count

jaccard_results = {}
for k, gram_type in k_values:
    pairs = list(combinations(documents.keys(), 2))
    for doc1, doc2 in pairs:
        set1 = kgram_data.get((doc1, k, gram_type), set())
        set2 = kgram_data.get((doc2, k, gram_type), set())
        similarity = jaccard_similarity(set1, set2)
        jaccard_results[(doc1, doc2, k, gram_type)] = similarity


# Min-Hashing Implementation
def minhash_signature(grams, num_hashes, m=10000):
    signature = []
    for i in range(num_hashes):
        hash_function = lambda x: (hash(x) + i) % m
        signature.append(min(hash_function(g) for g in grams))
    return signature


def estimated_jaccard(sig1, sig2):
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


t_values = [20, 60, 150, 300, 600]
minhash_results = {}

d1_grams = kgram_data.get(("D1", 3, 'char'), set())
d2_grams = kgram_data.get(("D2", 3, 'char'), set())

for t in t_values:
    sig1 = minhash_signature(d1_grams, t)
    sig2 = minhash_signature(d2_grams, t)
    minhash_results[t] = estimated_jaccard(sig1, sig2)

# Selecting the best t
print("\nB: Best t value based on accuracy vs. time tradeoff:")
for t, sim in minhash_results.items():
    print(f"t={t}: Estimated Jaccard = {sim:.4f}")


# LSH Implementation
def compute_lsh_params(t, tau):
    best_r, best_b = None, None
    min_diff = float('inf')
    for r in range(1, t + 1):
        if t % r == 0:
            b = t // r
            f_tau = 1 - (1 - tau ** b) ** r
            diff = abs(f_tau - 0.5)
            if diff < min_diff:
                min_diff = diff
                best_r, best_b = r, b
    return best_r, best_b


r, b = compute_lsh_params(160, 0.7)
print(f"Optimal LSH parameters: r = {r}, b = {b}")


def lsh_probability(s, b, r):
    return 1 - (1 - s ** b) ** r


similarity_values = [jaccard_results[(doc1, doc2, 3, 'char')] for doc1, doc2 in combinations(documents.keys(), 2)]
lsh_probs = [lsh_probability(s, b, r) for s in similarity_values]

for ((doc1, doc2), prob) in zip(combinations(documents.keys(), 2), lsh_probs):
    print(f"LSH Probability({doc1}, {doc2}) > 0.7: {prob:.4f}")

import numpy as np
import matplotlib.pyplot as plt

t = 160
tau = 0.7


def find_optimal_rb(t, tau):
    best_r, best_b, best_sep = None, None, float('inf')

    for b in range(1, t + 1):
        if t % b == 0:
            r = t // b

            separation = abs(1 - (1 - tau ** b) ** r - 0.5)

            if separation < best_sep:
                best_r, best_b, best_sep = r, b, separation

    return best_r, best_b


r_opt, b_opt = find_optimal_rb(t, tau)


def s_curve(s, r, b):
    return 1 - (1 - s ** b) ** r


s_values = np.linspace(0, 1, 100)
y_values = [s_curve(s, r_opt, b_opt) for s in s_values]

plt.figure(figsize=(8, 5))
plt.plot(s_values, y_values, label=f"S-Curve (r={r_opt}, b={b_opt})", color='blue')
plt.axvline(tau, color='red', linestyle='dashed', label=f"τ = {tau}")
plt.xlabel('Jaccard Similarity')
plt.ylabel('Probability of Candidate Pair')
plt.title('LSH S-Curve for Optimized (r, b)')
plt.legend()
plt.grid()
plt.show()

similarities = jaccard_results.values()
probabilities = [s_curve(s, r_opt, b_opt) for s in similarities]

# print(f"Optimal values: r = {r_opt}, b = {b_opt}")
# print("Probabilities of detecting document pairs above τ:")
# for i, (s, p) in enumerate(zip(similarities, probabilities)):
#     print(f"Pair {i + 1}: Jaccard = {s}, Probability = {p:.4f}")
