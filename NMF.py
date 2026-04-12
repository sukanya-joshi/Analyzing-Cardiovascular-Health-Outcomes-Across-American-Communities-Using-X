import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import CoherenceModel

# ============================================================
# NMF ON 100 TOPICS ONLY
# Assumes these already exist from earlier pipeline:
#   df["message"]
#   texts
#   dictionary
#   corpus
# ============================================================

N_TOPICS = 100
TOP_N_WORDS = 10

# Faster / lighter settings for large data
MAX_FEATURES = 20000
MAX_ITER = 150

# Use the already-cleaned messages from the same pipeline
documents = df["message"].astype(str).tolist()

# ============================================================
# TF-IDF
# ============================================================
vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=5,
    stop_words="english",
    max_features=MAX_FEATURES
)

tfidf = vectorizer.fit_transform(documents)

try:
    feature_names = np.array(vectorizer.get_feature_names_out())
except AttributeError:
    feature_names = np.array(vectorizer.get_feature_names())

print(f"TF-IDF matrix shape: {tfidf.shape}")

# ============================================================
# FIT NMF
# ============================================================
nmf_model = NMF(
    n_components=N_TOPICS,
    init="nndsvd",
    random_state=42,
    max_iter=MAX_ITER
)

W = nmf_model.fit_transform(tfidf)
H = nmf_model.components_

print(f"NMF fit complete: {N_TOPICS} topics")

# ============================================================
# EXTRACT TOPIC WORDS
# ============================================================
topics_raw = []
for topic_weights in H:
    top_indices = np.argsort(topic_weights)[::-1][:TOP_N_WORDS]
    topic_words = feature_names[top_indices].tolist()
    topics_raw.append(topic_words)

print(f"NMF topics loaded: {len(topics_raw):,}")

# ============================================================
# FILTER TOPICS TO WORDS IN GENSIM DICTIONARY
# so coherence uses the same vocab space as your earlier pipeline
# ============================================================
vocab = set(dictionary.token2id.keys())
topics = [[w for w in topic if w in vocab] for topic in topics_raw]
topics = [topic for topic in topics if len(topic) >= 2]

print(f"NMF topics retained after vocab filtering: {len(topics):,}")

if not topics:
    raise ValueError(
        "All NMF topics were removed after filtering to dictionary vocabulary. "
        "Your NMF tokenization and coherence tokenization may not match."
    )

topic_lengths = [len(t) for t in topics]
print(f"Min topic length after filtering: {min(topic_lengths)}")
print(f"Max topic length after filtering: {max(topic_lengths)}")

# Optional: inspect first few topics
for i, topic in enumerate(topics[:5]):
    print(f"Topic {i}: {topic}")

# ============================================================
# COMPUTE COHERENCE METRICS
# ============================================================
results = {
    "c_v": CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    ).get_coherence(),

    "c_uci": CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_uci"
    ).get_coherence(),

    "c_npmi": CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_npmi"
    ).get_coherence(),

    "u_mass": CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence="u_mass"
    ).get_coherence(),
}

# ============================================================
# FORMAT RESULTS
# ============================================================
results_df = pd.DataFrame([results], index=["NMF_100"]).reset_index()
results_df = results_df.rename(columns={"index": "model"})

print("\nNMF coherence results:")
display(results_df)

# # Optional: save results
# results_df.to_csv("nmf_100_coherence_scores.csv", index=False)
# print("\nSaved to nmf_100_coherence_scores.csv")
