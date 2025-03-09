import math
from itertools import combinations


def s_curve(x, a=1, b=1):
    return 1 / (1 + math.exp(-a * (x - b)))


def read_movielens_data(file_path):
    user_movies = {}
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            user, movie, _, _ = line.strip().split('\t')
            user_movies.setdefault(user, set()).add(movie)
    return user_movies


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def minhash_signature(items, num_hashes, m=10000):
    signature = []
    for i in range(num_hashes):
        hash_function = lambda x: (hash(x) + i) % m
        signature.append(min(hash_function(g) for g in items))
    return signature


def estimated_jaccard(sig1, sig2):
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


# Load MovieLens dataset
movie_data_path = "ml-100k/u.data"
user_movies = read_movielens_data(movie_data_path)

exact_jaccard = {}
user_pairs = list(combinations(user_movies.keys(), 2))
for user1, user2 in user_pairs:
    sim = jaccard_similarity(user_movies[user1], user_movies[user2])
    if sim >= 0.5:
        exact_jaccard[(user1, user2)] = sim
print(f"Exact Jaccard pairs (≥ 0.5): {len(exact_jaccard)}")

hash_counts = [50, 100, 200]
minhash_results = {}
false_positives, false_negatives = {}, {}

for t in hash_counts:
    user_signatures = {user: minhash_signature(movies, t) for user, movies in user_movies.items()}
    approx_jaccard = {}
    for user1, user2 in user_pairs:
        sim = estimated_jaccard(user_signatures[user1], user_signatures[user2])
        if sim >= 0.5:
            approx_jaccard[(user1, user2)] = sim

    false_pos = sum(1 for pair in approx_jaccard if pair not in exact_jaccard)
    false_neg = sum(1 for pair in exact_jaccard if pair not in approx_jaccard)

    false_positives[t] = false_pos / 5
    false_negatives[t] = false_neg / 5

    print(f"MinHash t={t}: False Positives = {false_pos}, False Negatives = {false_neg}")


# Locality Sensitive Hashing (LSH)
def lsh_candidate_pairs(signatures, r, b):
    bands = [{} for _ in range(b)]
    for user, sig in signatures.items():
        for i in range(b):
            band = tuple(sig[i * r:(i + 1) * r])
            bands[i].setdefault(band, set()).add(user)

    candidate_pairs = set()
    for band in bands:
        for bucket in band.values():
            for pair in combinations(bucket, 2):
                candidate_pairs.add(pair)
    return candidate_pairs


lsh_params = {
    50: (5, 10),
    100: (5, 20),
    200: [(5, 40), (10, 20)]
}

for t, params in lsh_params.items():
    user_signatures = {user: minhash_signature(movies, t) for user, movies in user_movies.items()}
    if isinstance(params, list):
        for r, b in params:
            candidates = lsh_candidate_pairs(user_signatures, r, b)
            false_pos = sum(1 for pair in candidates if pair not in exact_jaccard)
            false_neg = sum(1 for pair in exact_jaccard if pair not in candidates)
            print(f"LSH t={t}, r={r}, b={b}: False Positives = {false_pos}, False Negatives = {false_neg}")
    else:
        r, b = params
        candidates = lsh_candidate_pairs(user_signatures, r, b)
        false_pos = sum(1 for pair in candidates if pair not in exact_jaccard)
        false_neg = sum(1 for pair in exact_jaccard if pair not in candidates)
        print(f"LSH t={t}, r={r}, b={b}: False Positives = {false_pos}, False Negatives = {false_neg}")
