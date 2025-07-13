import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image, ImageFile
from datasets import load_dataset
from transformers import set_seed

from vlm2vec_utils.model import MMEBModel
from vlm2vec_utils.arguments import ModelArguments
from vlm2vec_utils.model_utils import load_processor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_image_token(model_backbone):
    return "<|image_pad|>" if model_backbone == "qwen2_vl" else "<image>"


def encode(model, processor, image=None, text=None, device="cpu"):
    if image:
        image = image.convert("RGB").resize((1344, 1344))
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return model(tgt=inputs)["tgt_reps"].squeeze().cpu()


def process_dataset(args, model_args):
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    processor = load_processor(model_args)
    model = MMEBModel.load(model_args)
    model = model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()

    dataset_query = load_dataset(args.dataset_name, name=args.subset_name, split="query")
    dataset_corpus = load_dataset(args.dataset_name, name=args.subset_name, split="corpus")
    results = []

    image_token = get_image_token(model_args.model_backbone)
    # queries
    for i, sample in enumerate(tqdm(dataset_query, desc=f"Processing queries{args.subset_name}")):
        query_id = sample["query_id"]
        qry_caption = sample["text"]
        qry_txt_emb = encode(model, processor, text=f"Retrieve a relevant item that represents: {qry_caption}\n", device=device)

        results.append({
            "query_id": query_id,
            "text_embedding": qry_txt_emb,
        })
    torch.save(results, args.query_save_path)
    
    # corpus
    results = []
    for i, sample in enumerate(tqdm(dataset_corpus, desc=f"Processing {args.dataset_name}")):
        
        corpus_id = sample["corpus_id"]
        tgt_caption = sample["text"]
        image_path = os.path.join(args.image_root, os.path.basename(sample["image"]))
        image = Image.open(image_path)
        
        tgt_txt_emb = encode(model, processor, text=tgt_caption, device=device)
        tgt_img_emb = encode(model, processor, image=image, text=f"{image_token}\nRepresent the given image\n", device=device)
        mixed_tgt_emb = encode(model, processor, image=image, text=f"{image_token}\nRepresent the given image with related text information: {tgt_caption}\n", device=device)

        results.append({
            "corpus_id": corpus_id,
            "image_embedding": tgt_img_emb,
            "text_embedding": tgt_txt_emb,
            "vlm_mixed_embedding": mixed_tgt_emb
        })

    torch.save(results, args.corpus_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--subset_name", type=str, default='MSCOCO')
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--query_save_path", type=str, required=True)
    parser.add_argument("--corpus_save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--model_backbone", type=str, default="qwen2_vl")
    parser.add_argument("--lora", action="store_true")

    args = parser.parse_args()

    model_args = ModelArguments(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        pooling=args.pooling,
        normalize=args.normalize,
        model_backbone=args.model_backbone,
        lora=args.lora,
    )

    process_dataset(args, model_args)
