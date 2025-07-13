import torch
import numpy as np
import random
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_query_embeddings(path):
    data = torch.load(path)
    return {
        d["query_id"]: torch.nn.functional.normalize(d["text_embedding"], dim=0).to(DEVICE)
        for d in data if d["text_embedding"] is not None
    }

def load_corpus_embeddings(path):
    data = torch.load(path)
    corpus = {}
    for d in data:
        if d["text_embedding"] is not None and d["image_embedding"] is not None:
            corpus[d["corpus_id"]] = {
                "text": torch.nn.functional.normalize(d["text_embedding"], dim=0).to(DEVICE),
                "image": torch.nn.functional.normalize(d["image_embedding"], dim=0).to(DEVICE),
            }
    return corpus

def get_recall1_alpha_blend(query_embed_path, corpus_embed_path, query_mean_path, corpus_text_mean_path, corpus_image_mean_path,
                             seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load and truncate data
    query_dict = load_query_embeddings(query_embed_path)
    corpus_dict = load_corpus_embeddings(corpus_embed_path)
    query_mean = torch.load(query_mean_path).to(DEVICE)
    corpus_text_mean = torch.load(corpus_text_mean_path).to(DEVICE)
    corpus_image_mean = torch.load(corpus_image_mean_path).to(DEVICE)
    query_ids = list(query_dict.keys())
    corpus_ids = list(corpus_dict.keys())

    query_matrix = torch.stack([query_dict[qid] for qid in query_ids])
    corpus_text_matrix = torch.stack([corpus_dict[cid]["text"] for cid in corpus_ids])
    corpus_image_matrix = torch.stack([corpus_dict[cid]["image"] for cid in corpus_ids])

    alphas = np.arange(0.0, 1.01, 0.05)
    recall1_normal, recall1_GR = [], []

    for alpha in tqdm(alphas, desc="Alpha Sweep"):
        corpus_blend = alpha * corpus_text_matrix + (1 - alpha) * corpus_image_matrix
        corpus_blend_GR = alpha * (corpus_text_matrix - corpus_text_mean) + (1 - alpha) * (corpus_image_matrix - corpus_image_mean)
        corpus_norm = torch.nn.functional.normalize(corpus_blend, p=2, dim=1)
        corpus_GR = torch.nn.functional.normalize(corpus_blend_GR, p=2, dim=1)
        query_norm = torch.nn.functional.normalize(query_matrix, p=2, dim=1)

        sims = torch.matmul(query_norm, corpus_norm.T)
        top1 = torch.argmax(sims, dim=1)
        recall = [
            1 if query_ids[i] == corpus_ids[top1[i]] else 0
            for i in range(len(query_ids))
        ]
        recall1_normal.append(np.mean(recall))

        # Gap Removed (GR) version
        query_GR = torch.nn.functional.normalize(query_norm - query_mean, p=2, dim=1)
        sims_GR = torch.matmul(query_GR, corpus_GR.T)
        top1_GR = torch.argmax(sims_GR, dim=1)
        recall_c = [
            1 if query_ids[i] == corpus_ids[top1_GR[i]] else 0
            for i in range(len(query_ids))
        ]
        recall1_GR.append(np.mean(recall_c))

    return alphas, recall1_normal, recall1_GR

# === Usage ===
# Replace the following paths with your actual files
# alphas, r1, r1_gr = get_recall1_alpha_blend(
#     query_embed_path="mmscoco_query_embeddings.pt",
#     corpus_embed_path="mmscoco_corpus_embeddings.pt",
#     query_mean_path="query_clip_mean.pt",
#     corpus_text_mean_path="corpus_clip_text_mean.pt",
#     corpus_image_mean_path="corpus_clip_image_mean.pt"
# )
