import torch
import numpy as np
import random
from datasets import load_dataset
from sklearn.metrics import ndcg_score

ALPHA = 0.5


def load_query_embeddings(path):
    data = torch.load(path)
    return {d["query_id"]: torch.nn.functional.normalize(d["text_embedding"], dim=0)
            for d in data if d["text_embedding"] is not None}

def load_corpus_embeddings(path):
    data = torch.load(path)
    corpus = []
    for d in data:
        if d["text_embedding"] is not None and d["image_embedding"] is not None:
            corpus.append({
                "corpus_id": d["corpus_id"],
                "text": torch.nn.functional.normalize(d["text_embedding"], dim=0),
                "image": torch.nn.functional.normalize(d["image_embedding"], dim=0),
            })
    return corpus

def load_qrels():
    qrels = {}
    for row in load_dataset("mixed-modality-search/MixBench2025", name="MSCOCO", split="qrel"):
        qrels.setdefault(row["query_id"], {})[row["corpus_id"]] = int(row["score"])
    return qrels

def compute_ndcg(query_embeds, corpus, qrels, alpha, k=10, use_mean=False,
                 qry_text_mean=None, doc_text_mean=None, doc_image_mean=None):
    corpus_embeds = []
    for d in corpus:
        if d["mode"] == "text":
            emb = d["text"]
            if use_mean:
                emb = emb - doc_text_mean
        elif d["mode"] == "image":
            emb = d["image"]
            if use_mean:
                emb = emb - doc_image_mean
        else:  # mixed
            emb = alpha * d["text"] + (1 - alpha) * d["image"]
            if use_mean:
                mix_mean = alpha * doc_text_mean + (1 - alpha) * doc_image_mean
                emb = emb - mix_mean
        corpus_embeds.append(torch.nn.functional.normalize(emb, dim=0))

    Q = []
    for qid in query_embeds:
        emb = query_embeds[qid]
        if use_mean:
            emb = emb - qry_text_mean
        Q.append(torch.nn.functional.normalize(emb, dim=0))

    Q = torch.stack(Q)
    D = torch.stack(corpus_embeds)
    sims = torch.matmul(Q, D.T)

    query_ids = list(query_embeds.keys())
    corpus_ids = [d["corpus_id"] for d in corpus]
    ndcgs = []
    for i, qid in enumerate(query_ids):
        top_idx = torch.topk(sims[i], k=k).indices
        top_cids = [corpus_ids[j] for j in top_idx.tolist()]
        rel = [qrels.get(qid, {}).get(cid, 0) for cid in top_cids]
        ndcgs.append(ndcg_score([rel], [sims[i, top_idx].cpu().numpy()], k=k))
    return np.mean(ndcgs)

# === MAIN ===
query_embeddings = load_query_embeddings("mmscoco_query_embeddings.pt")
corpus = load_corpus_embeddings("mmscoco_corpus_embeddings.pt")
qrels = load_qrels()

# 1:1:1 random sample text image and mixed
random.seed(42)
n = len(corpus) // 3
perm = random.sample(range(len(corpus)), len(corpus))
for i in range(len(corpus)):
    corpus[perm[i]]["mode"] = ["text", "image", "mixed"][i % 3]

# Precompute means：
qry_text_mean = "path/to/your/qry_text_mean"
doc_text_mean = "path/to/your/doc_text_mean"
doc_image_mean = "path/to/your/doc_image_mean"

results = []
alpha = ALPHA
score_no = compute_ndcg(query_embeddings, corpus, qrels, alpha, use_mean=False)
score_gr = compute_ndcg(query_embeddings, corpus, qrels, alpha, use_mean=True,
                         qry_text_mean=qry_text_mean,
                         doc_text_mean=doc_text_mean,
                         doc_image_mean=doc_image_mean)
results.append((alpha, score_no, score_gr))
print(f"{alpha:.2f}\t{score_no:.4f}\t\t{score_gr:.4f}")

