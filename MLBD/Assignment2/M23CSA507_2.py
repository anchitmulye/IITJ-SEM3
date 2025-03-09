import os
from itertools import combinations


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

# Compute k-grams and Jaccard similarity
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

# Question A
print("\nA: Approximate jaccard similarity:")
for t in t_values:
    sig1 = minhash_signature(d1_grams, t)
    sig2 = minhash_signature(d2_grams, t)
    minhash_results[t] = estimated_jaccard(sig1, sig2)
    print(f"MinHash Jaccard(D1, D2) with t={t}: {minhash_results[t]:.4f}")

# Question B
print("\nB: Best t value based on accuracy vs. time tradeoff:")
for t, sim in minhash_results.items():
    print(f"t={t}: Estimated Jaccard = {sim:.4f}")

