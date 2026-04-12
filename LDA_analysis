import pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# =========================
# CONFIG
# =========================
MESSAGES_PATH = "cleaned_acp_600k.csv"
TOP_N_WORDS = 10
NO_BELOW = 5
NO_ABOVE = 0.5

TOPIC_FILES = {
    "LDA_50": "lda.wordGivenTopic_cleaned_50.csv",
    "LDA_100": "lda.wordGivenTopic_cleaned_100.csv",
    "LDA_200": "lda.wordGivenTopic_cleaned_200.csv",
}

# =========================
# LOAD CLEANED MESSAGES
# =========================
df = pd.read_csv(MESSAGES_PATH, usecols=["message"]).dropna(subset=["message"])
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"] != ""].copy()

# =========================
# TOKENIZE ALREADY-CLEANED TEXT
# =========================
def tokenize_cleaned_message(text):
    text = str(text).lower().strip()
    raw_tokens = text.split()
    tokens = []

    for tok in raw_tokens:
        tok = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", tok)
        if tok:
            tokens.append(tok)

    return tokens

texts = [tokenize_cleaned_message(msg) for msg in df["message"]]
texts = [t for t in texts if t]

print(f"Loaded messages: {len(df):,}")
print(f"Non-empty tokenized documents: {len(texts):,}")

# =========================
# BUILD DICTIONARY + CORPUS
# =========================
dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Dictionary size after filtering: {len(dictionary):,}")

# =========================
# LOAD TOPICS
# Expected format:
# topic_id,word1,word1_score,word2,word2_score,...
# =========================
def normalize_topic_word(word):
    word = str(word).strip().lower()
    word = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", word)
    return word

def load_topics(path, top_n=10):
    topics = []

    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            word_score_parts = parts[1:]  # skip topic_id

            words = []
            for i in range(0, len(word_score_parts), 2):
                word = normalize_topic_word(word_score_parts[i])
                if word:
                    words.append(word)
                if len(words) >= top_n:
                    break

            if words:
                topics.append(words)

    return topics

# =========================
# FILTER TOPICS TO WORDS IN DICTIONARY
# =========================
def filter_topics_to_vocab(topics_raw, dictionary):
    vocab = set(dictionary.token2id.keys())
    topics = [[w for w in topic if w in vocab] for topic in topics_raw]
    topics = [topic for topic in topics if len(topic) >= 2]
    return topics

# =========================
# COMPUTE COHERENCE METRICS
# =========================
def compute_metrics(topics, texts, dictionary, corpus):
    return {
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

# =========================
# RUN ALL THREE LDA MODELS
# =========================
all_results = []

for model_name, topic_path in TOPIC_FILES.items():
    print(f"\nProcessing {model_name} from {topic_path}")

    topics_raw = load_topics(topic_path, top_n=TOP_N_WORDS)
    print(f"Loaded topics: {len(topics_raw):,}")

    topics = filter_topics_to_vocab(topics_raw, dictionary)
    print(f"Topics retained after vocab filtering: {len(topics):,}")

    if not topics:
        raise ValueError(
            f"All topics were removed for {model_name} after filtering to dictionary vocabulary."
        )

    metrics = compute_metrics(topics, texts, dictionary, corpus)
    metrics["model"] = model_name
    all_results.append(metrics)

# =========================
# SHOW RESULTS
# =========================
results_df = pd.DataFrame(all_results)
results_df["n_topics"] = results_df["model"].str.extract(r"(\d+)").astype(int)
results_df = results_df.sort_values("n_topics").reset_index(drop=True)

results_df = results_df[["model", "c_v", "c_uci", "c_npmi", "u_mass"]]

print("\nCoherence results:")
display(results_df)

# Optional save
results_df.to_csv("lda_coherence_scores_50_100_200.csv", index=False)
print("\nSaved to lda_coherence_scores_50_100_200.csv")
