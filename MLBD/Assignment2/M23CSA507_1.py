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

# k-grams count
kgram_counts = {}
for (doc, k, gram_type), kgram_set in kgram_data.items():
    count = len(kgram_set)
    kgram_counts[(doc, k, gram_type)] = count

# Jaccard similarities
jaccard_results = {}
for k, gram_type in k_values:
    pairs = list(combinations(documents.keys(), 2))
    for doc1, doc2 in pairs:
        set1 = kgram_data.get((doc1, k, gram_type), set())
        set2 = kgram_data.get((doc2, k, gram_type), set())
        similarity = jaccard_similarity(set1, set2)
        jaccard_results[(doc1, doc2, k, gram_type)] = similarity

# Question A
print("\nA: Distinct k-grams for each document:")
for key, value in kgram_counts.items():
    print(f"{key[0]} ({key[1]}-gram, {key[2]}): {value} unique k-grams")

# Question B
print("\nB: Jaccard similarity between document pairs:")
for key, value in jaccard_results.items():
    print(f"Jaccard({key[0]}, {key[1]}) [{key[2]}-gram, {key[3]}]: {value:.4f}")

