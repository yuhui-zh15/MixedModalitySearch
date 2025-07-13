# embedding_utils.py

import torch
import torch.nn.functional as F
from PIL import Image
import os

def get_clip_embeddings(texts=None, image_paths=None, batch_size=16,
                        model=None, model_type="clip",
                        processor=None, tokenizer=None,
                        device="cuda"):
    
    assert texts or image_paths, "At least one of texts or image_paths must be provided."
    assert model is not None, "You must provide a model."
    results = []
    total = max(len(texts) if texts else 0, len(image_paths) if image_paths else 0)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size] if texts else None
        batch_image_paths = image_paths[i:i + batch_size] if image_paths else None

        # images
        batch_images = []
        valid_image_indices = []
        if batch_image_paths:
            for j, path in enumerate(batch_image_paths):
                if path and os.path.exists(path):
                    img = Image.open(path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    batch_images.append(img)
                    valid_image_indices.append(j)
                else:
                    batch_images.append(None)

        # Text embeddings
        text_embeds = [None] * (len(batch_texts) if batch_texts else 0)
        if batch_texts:
            if model_type in ["clip", "siglip"]:
                text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    text_out = model.get_text_features(**text_inputs)
            elif model_type == "openclip":
                assert tokenizer is not None, "tokenizer is required for openclip"
                tokenized = tokenizer(batch_texts).to(device)
                with torch.no_grad():
                    text_out = model.encode_text(tokenized, normalize=False).squeeze()
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            text_out = text_out.cpu()
            for j in range(len(text_out)):
                text_embeds[j] = text_out[j]

        # Image embeddings
        image_embeds = [None] * (len(batch_images) if batch_image_paths else 0)
        if valid_image_indices:
            valid_images = [batch_images[j] for j in valid_image_indices]
            if model_type in ["clip", "siglip"]:
                image_inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_out = model.get_image_features(**image_inputs)
            elif model_type == "openclip":
                image_inputs = torch.stack([processor(img) for img in valid_images]).to(device)
                with torch.no_grad():
                    image_out = model.encode_image(image_inputs, normalize=False).squeeze()
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            image_out = F.normalize(image_out, dim=-1).cpu()
            for idx, j in enumerate(valid_image_indices):
                image_embeds[j] = image_out[idx]

        # Collect results 
        for j in range(max(len(text_embeds), len(image_embeds))):
            t_emb = text_embeds[j] if j < len(text_embeds) else None
            i_emb = image_embeds[j] if j < len(image_embeds) else None
            results.append((t_emb, i_emb))

    return results
