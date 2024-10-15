import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from transformers import CLIPModel

from .components.combiner_network import (
    Combiner_basic,
    Combiner_cross_attention,
    Combiner_transformer,
    Combiner_transformer2,
)


def get_clip(trainable=False):
    """Get CLIP model from Hugging Face's Transformers library."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Set parameters to be non-trainable
    if not trainable:
        for param in model.parameters():  # type: ignore
            param.requires_grad = False
    return model


class CDC(nn.Module):
    """CLIP-based Deep Clustering (CDC) module."""

    def __init__(
        self,
        clip_trainable: bool = False,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        # Frozen CLIP as feature extractor
        self.clip = get_clip(clip_trainable)

        # Vision Model
        self.clip_vm = self.clip.vision_model  # type: ignore
        self.clip_vproj = self.clip.visual_projection  # type: ignore

        # Textual Model
        self.clip_tm = self.clip.text_model  # type: ignore
        self.clip_tproj = self.clip.text_projection  # type: ignore

        # Identity function for now
        self.label_encoder = nn.Identity()

        # Combiner network to combine text and label features
        self.combiner = Combiner_transformer2(
            512, 512, d_model, num_heads=nhead, num_layers=num_layers
        )

    def encode_img(self, images):
        # Extract image features
        img_output = self.clip_vm(**images)
        img_emb = self.clip_vproj(img_output.pooler_output)  # CLS token
        img_full = self.clip_vproj(img_output.last_hidden_state)  # Full image embeddings

        return img_emb, img_full

    def encode_txt(self, texts):
        # Extract text features
        txt_output = self.clip_tm(**texts)
        txt_emb = self.clip_tproj(txt_output.pooler_output)  # CLS token
        txt_full = self.clip_tproj(txt_output.last_hidden_state)  # Full text embeddings

        return txt_emb, txt_full

    def encode_img_txt(self, images, texts):
        # Extract image and text features

        img_emb, img_full = self.encode_img(images)
        txt_emb, txt_full = self.encode_txt(texts)

        return img_emb, txt_emb, img_full, txt_full

    def combine(self, txt_emb, txt_full, labels):
        # Encode the labels
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)
        comb_emb = self.combiner(txt_emb, txt_full, lbl_emb)  # (batch_size, 512)

        return comb_emb

    def forward(self, images, texts, labels):
        # Extract image and text features

        img_emb, txt_emb, _, txt_full = self.encode_img_txt(images, texts)

        # Encode the labels
        lbl_emb = self.label_encoder(labels)  # (batch_size, 512)

        # Combine text and label features
        comb_emb = self.combiner(txt_emb, txt_full, lbl_emb)  # (batch_size, 512)

        return (
            img_emb,
            txt_emb,
            txt_full,
            lbl_emb,
            comb_emb,
        )  # For now we only need img_emb and comb_emb to calculate the loss


def main():
    # Load pretrained model
    import requests
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = torch.randn(2, 512)

    image_input = processor(images=[image, image], return_tensors="pt")
    text_input = processor(["a photo of a cat", "a photo of a dog"], return_tensors="pt")

    model = CDC()

    img_emb, txt_emb, lbl_emb, comb_emb = model(image_input, text_input, label)
    print(img_emb.shape, txt_emb.shape, lbl_emb.shape, comb_emb.shape)


if __name__ == "__main__":
    main()
