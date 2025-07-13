import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from embedding_utils import get_clip_embeddings
import open_clip
from transformers import SiglipProcessor, SiglipModel


def process_split(split_name, id_field, output_path,
                  model, model_type, processor=None, tokenizer=None,
                  image_root=None, batch_size=16, subset="MSCOCO", device="cuda"):
    dataset = load_dataset("mixed-modality-search/MixBench25", name=subset, split=split_name)
    ids, texts, image_paths = [], [], []

    for sample in dataset:
        ids.append(sample[id_field])
        texts.append(sample.get("text", None))
        image_rel_path = sample.get("image", None)
        full_image_path = os.path.join(image_root, image_rel_path) if image_rel_path and image_root else None
        image_paths.append(full_image_path if full_image_path and os.path.exists(full_image_path) else None)

    embeddings = get_clip_embeddings(
        texts=texts,
        image_paths=image_paths,
        batch_size=batch_size,
        model=model,
        model_type=model_type,
        processor=processor,
        tokenizer=tokenizer,
        device=device
    )

    output = []
    for i in range(len(ids)):
        output.append({
            f"{id_field}": ids[i],
            "text_embedding": embeddings[i][0],
            "image_embedding": embeddings[i][1]
        })

    torch.save(output, output_path)
    print(f"Saved {len(output)} samples to {output_path}")


if __name__ == "__main__":
    
    subset = "MSCOCO"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_type = "clip"  # or "siglip", or "openclip"
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = None 
    
    # siglip
    
    # model_type = "siglip"  # or "siglip", or "openclip"
    # model_id = "google/siglip-so400m-patch14-384"
    # model = SiglipModel.from_pretrained(model_id).to(device)
    # processor = SiglipProcessor.from_pretrained(model_id)
    # tokenizer = None 
    
    # openclip
    
    # model_type = "siglip"
    # model_id = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    # model, processor = open_clip.create_model_from_pretrained(model_id)
    # tokenizer = open_clip.get_tokenizer(model_id)
    
    # === Process query split ===
    process_split(
        split_name="query",
        id_field="query_id",
        output_path=f"./mmscoco_query_embeddings.pt",
        image_root="MixBench_img/MSCOCO/images",
        batch_size=16,
        subset=subset,
        model=model,
        model_type=model_type,
        processor=processor,
        tokenizer=tokenizer,
        device=device
    )

    # === Process corpus split ===
    process_split(
        split_name="corpus",
        id_field="corpus_id",
        output_path=f"./mmscoco_corpus_embeddings.pt",
        image_root="MixBench_img/MSCOCO/images",
        batch_size=16,
        subset=subset,
        model=model,
        model_type=model_type,
        processor=processor,
        tokenizer=tokenizer,
        device=device
    )
