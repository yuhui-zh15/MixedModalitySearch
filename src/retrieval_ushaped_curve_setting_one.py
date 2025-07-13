import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import ndcg_score
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def get_ndcg(query_emb_path, corpus_emb_path, qrels_path,
             model_name, q_mean, t_mean, i_mean, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    query_list = torch.load(query_emb_path)
    query_ids = [entry['id'] for entry in query_list]
    query_matrix = F.normalize(torch.stack([entry['text_embedding'] for entry in query_list]), dim=-1)
    query_emb = {
        qid: emb for qid, emb in zip(query_ids, query_matrix)
    }
    corpus_list = torch.load(corpus_emb_path)
    corpus_ids = [entry['id'] for entry in corpus_list]
    corpus_text_matrix = F.normalize(torch.stack([entry['text_embedding'] for entry in corpus_list]), dim=-1).float().numpy()
    corpus_img_matrix = F.normalize(torch.stack([entry['img_embedding'] for entry in corpus_list]), dim=-1).float().numpy()

    q_mean = q_mean.squeeze().numpy() if isinstance(q_mean, torch.Tensor) else q_mean
    t_mean = t_mean.squeeze().numpy() if isinstance(t_mean, torch.Tensor) else t_mean
    i_mean = i_mean.squeeze().numpy() if isinstance(i_mean, torch.Tensor) else i_mean

    qrels = defaultdict(dict)
    with open(qrels_path, "r") as f:
        next(f)
        for line in f:
            qid, cid, score = line.strip().split()
            qrels[qid][cid] = int(score)

    ratios = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.]
    ndcg10_list, ndcg100_list = [], []
    ndcg10_gr, ndcg100_gr = [], []
    ndcg10_opt, ndcg100_opt = [], []

    for ratio in tqdm(ratios, desc=f"Testing Ratios for {model_name}"):
        num_replace = int(len(corpus_list) * ratio)
        indices_to_replace = set(random.sample(range(len(corpus_list)), num_replace))

        mixed_corpus = np.stack([
            corpus_img_matrix[i] if i in indices_to_replace else corpus_text_matrix[i]
            for i in range(len(corpus_list))
        ])
        mixed_corpus_gr = np.stack([
            corpus_img_matrix[i] - i_mean if i in indices_to_replace else corpus_text_matrix[i] - t_mean
            for i in range(len(corpus_list))
        ])

        ndcg10_all, ndcg100_all = [], []
        ndcg10_all_gr, ndcg100_all_gr = [], []
        ndcg10_opt_all, ndcg100_opt_all = [], []

        for qid, qvec in query_emb.items():
            if qid not in qrels: continue
            qvec = qvec.float().numpy()
            qvec_gr = qvec - q_mean
            qnorm, qnorm_gr = np.linalg.norm(qvec), np.linalg.norm(qvec_gr)
            c_norm = np.linalg.norm(mixed_corpus, axis=1)
            c_norm_gr = np.linalg.norm(mixed_corpus_gr, axis=1)

            sims = mixed_corpus @ qvec / (qnorm * c_norm)
            sims_gr = mixed_corpus_gr @ qvec_gr / (qnorm_gr * c_norm_gr)

            topk, topk_gr = np.argsort(sims)[::-1][:100], np.argsort(sims_gr)[::-1][:100]
            y_true = [qrels[qid].get(corpus_ids[i], 0) for i in topk]
            y_score = [sims[i] for i in topk]
            y_true_gr = [qrels[qid].get(corpus_ids[i], 0) for i in topk_gr]
            y_score_gr = [sims_gr[i] for i in topk_gr]

            ndcg10_all.append(ndcg_score([y_true], [y_score], k=10))
            ndcg100_all.append(ndcg_score([y_true], [y_score], k=100))
            ndcg10_all_gr.append(ndcg_score([y_true_gr], [y_score_gr], k=10))
            ndcg100_all_gr.append(ndcg_score([y_true_gr], [y_score_gr], k=100))

            # === Optimal simulation ranking (text-first) ===
            text_idx = [i for i in range(len(corpus_list)) if i not in indices_to_replace]
            image_idx = [i for i in range(len(corpus_list)) if i in indices_to_replace]

            sims_text = corpus_text_matrix[text_idx] @ qvec / (
                qnorm * np.linalg.norm(corpus_text_matrix[text_idx], axis=1))
            sims_image = corpus_img_matrix[image_idx] @ qvec / (
                qnorm * np.linalg.norm(corpus_img_matrix[image_idx], axis=1))

            sorted_idx = [text_idx[i] for i in np.argsort(sims_text)[::-1]] + \
                         [image_idx[i] for i in np.argsort(sims_image)[::-1]]
            topk_opt = sorted_idx[:100]
            y_true_opt = [qrels[qid].get(corpus_ids[i], 0) for i in topk_opt]
            y_score_opt = [1.0 - i / 100.0 for i in range(100)]

            ndcg10_opt_all.append(ndcg_score([y_true_opt], [y_score_opt], k=10))
            ndcg100_opt_all.append(ndcg_score([y_true_opt], [y_score_opt], k=100))

        ndcg10_list.append(np.mean(ndcg10_all))
        ndcg100_list.append(np.mean(ndcg100_all))
        ndcg10_gr.append(np.mean(ndcg10_all_gr))
        ndcg100_gr.append(np.mean(ndcg100_all_gr))
        ndcg10_opt.append(np.mean(ndcg10_opt_all))
        ndcg100_opt.append(np.mean(ndcg100_opt_all))

    return {
        'ratios': ratios,
        'ndcg10': ndcg10_list,
        'ndcg100': ndcg100_list,
        'ndcg10_gr': ndcg10_gr,
        'ndcg100_gr': ndcg100_gr,
        'ndcg10_opt': ndcg10_opt,
        'ndcg100_opt': ndcg100_opt
    }


def plot_ndcg_curve(data_path, save_path=None):
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return
    data = torch.load(data_path)
    if "ratios" not in data:
        print(f"[Error] Invalid data file, missing 'ratios'")
        return
    ratios = data["ratios"]

    plot_keys = [
        ("ndcg10", "NDCG@10 (Raw)", "o", "-"),
        ("ndcg100", "NDCG@100 (Raw)", "o", "--"),
        ("ndcg10_gr", "NDCG@10 (GR-CLIP)", "x", "-"),
        ("ndcg100_gr", "NDCG@100 (GR-CLIP)", "x", "--"),
        ("ndcg10_opt", "NDCG@10 (Optimal)", "s", "-."),
        ("ndcg100_opt", "NDCG@100 (Optimal)", "s", ":"),
    ]

    plt.figure(figsize=(8, 5))
    for key, label, marker, linestyle in plot_keys:
        if key in data:
            plt.plot(ratios, data[key], label=label, marker=marker, linestyle=linestyle)

    plt.xlabel("Image Replacement Ratio")
    plt.ylabel("NDCG Score")
    plt.title("NDCG vs. Replacement Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    query_emb_path = "clip_scifact_query_embeddings.pt"
    corpus_emb_path = "clip_scifact_corpus_embeddings.pt"
    qrels_path = "/path/to/qrels/test.tsv"
    q_mean = ".."
    t_mean = ".."
    i_mean = ".."

    result = get_ndcg(
        query_emb_path, corpus_emb_path, qrels_path,
        model_name="clip", q_mean=q_mean, t_mean=t_mean, i_mean=i_mean
    )
    # torch.save(result, "ndcg_clip_l_scifact_result.pt")
    # plot_ndcg_curve("ndcg_clip_l_scifact_result.pt")
