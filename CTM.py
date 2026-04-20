import pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.models.ctm import CombinedTM

# ============================================================
# SETTINGS
# ============================================================
MESSAGES_PATH = "/home/sukanyaj/cleaned_acp_600k.csv"

TOP_N_WORDS = 10
N_TOPICS = 100
CTM_EPOCHS = 3

# Better suited for a server than local laptop
ENCODER_MODEL = "paraphrase-distilroberta-base-v2"
CONTEXTUAL_SIZE = 768

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(MESSAGES_PATH, usecols=["message"]).dropna(subset=["message"])
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"] != ""].reset_index(drop=True)

if "message" not in df.columns:
    raise ValueError("Expected `df` to contain a 'message' column.")

raw_documents = df["message"].tolist()

if not raw_documents:
    raise ValueError("No non-empty raw documents found in the CSV.")

print(f"Loaded messages: {len(raw_documents):,}")

# ============================================================
# BUILD SHARED COHERENCE CORPUS
# This recreates the same style of evaluation corpus you used
# for LDA and NMF, so CTM coherence is directly comparable.
# ============================================================
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

print(f"Non-empty tokenized documents for coherence: {len(texts):,}")

dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Shared dictionary size after filtering: {len(dictionary):,}")

# ============================================================
# PREPROCESS FOR CTM TRAINING
# This is for CTM model fitting only.
# ============================================================
sp = WhiteSpacePreprocessing(raw_documents, stopwords_language="english")
processed = sp.preprocess()

preprocessed_docs = processed[0]
unpreprocessed_corpus = processed[1]
vocab = processed[2]

print(f"Non-empty CTM training documents: {len(preprocessed_docs):,}")
print(f"CTM preprocessing vocab size: {len(vocab):,}")

# ============================================================
# PREPARE CONTEXTUAL DATASET
# ============================================================
tp = TopicModelDataPreparation(ENCODER_MODEL)

training_dataset = tp.fit(
    text_for_contextual=unpreprocessed_corpus,
    text_for_bow=preprocessed_docs
)

print(f"CTM BoW vocabulary size: {len(tp.vocab):,}")

# ============================================================
# FIT CTM
# ============================================================
ctm_model = CombinedTM(
    bow_size=len(tp.vocab),
    contextual_size=CONTEXTUAL_SIZE,
    n_components=N_TOPICS,
    num_epochs=CTM_EPOCHS
)

ctm_model.fit(training_dataset)

print(f"CTM fit complete: {N_TOPICS} topics")

# ============================================================
# EXTRACT TOPIC WORDS
# ============================================================
topics_raw = ctm_model.get_topic_lists(TOP_N_WORDS)

print(f"CTM topics loaded: {len(topics_raw):,}")

# Filter CTM topic words to the shared gensim dictionary vocab
shared_vocab = set(dictionary.token2id.keys())
topics = [[w for w in topic if w in shared_vocab] for topic in topics_raw]
topics = [topic for topic in topics if len(topic) >= 2]

print(f"CTM topics retained after shared vocab filtering: {len(topics):,}")

if not topics:
    raise ValueError(
        "All CTM topics were removed after filtering to the shared dictionary vocabulary."
    )

topic_lengths = [len(t) for t in topics]
print(f"Min topic length after filtering: {min(topic_lengths)}")
print(f"Max topic length after filtering: {max(topic_lengths)}")

# Optional: inspect first 5 topics
for i, topic in enumerate(topics[:5]):
    print(f"Topic {i}: {topic}")

# ============================================================
# COMPUTE COHERENCE ON SHARED CORPUS
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
results_df = pd.DataFrame([results], index=["CTM_100"]).reset_index()
results_df = results_df.rename(columns={"index": "model"})

print("\nCTM coherence results:")
print(results_df)

# Optional save
results_df.to_csv("ctm_100_coherence_scores.csv", index=False)
print("\nSaved to ctm_100_coherence_scores.csv")
