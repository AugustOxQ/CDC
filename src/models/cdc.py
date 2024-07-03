import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torchvision
from transformers import CLIPModel
from torch.autograd import Variable

from .components.combiner_network import Combiner

def get_clip(trainable=False):
    """Get CLIP model from Hugging Face's Transformers library."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Set parameters to be non-trainable
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False
    return model

def l2norm(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)


class TransformerEncoder(nn.Module):
    """Transformer encoder module as used in BERT, etc. Used for encoding extra text features."""

    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super(TransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, enable_nested_tensor = False)

    def forward(self, src, mask=None):
        output = self.transformer_encoder(src, mask)
        return output


class CDC(nn.Module):
    """CLIP-based Deep Clustering (CDC) module"""

    def __init__(self, clip_trainable=False, d_model=512, nhead=8, num_layers=4):
        super(CDC, self).__init__()
        # Frozen CLIP as feature extractor
        self.clip = get_clip(clip_trainable)
        
        # The label encoder is a transformer encoder
        # self.label_encoder = TransformerEncoder(d_model, nhead, num_layers)
        # Identity function for now
        self.label_encoder = nn.Identity()
        
        # Combiner network to combine text and label features
        self.combiner = Combiner(512, 512, 512)
        
    def encode_img(self, images):
        # Extract image features
        img_emb = self.clip.get_image_features(**images)
        return img_emb
    
    def encode_txt(self, texts):
        # Extract text features
        txt_emb = self.clip.get_text_features(**texts)
        return txt_emb
        
    def encode_img_txt(self, images, texts):
        # Extract image and text features
        img_emb = self.clip.get_image_features(**images)
        txt_emb = self.clip.get_text_features(**texts)
        
        return img_emb, txt_emb
    
    def combine_raw(self, texts, labels):
        
        txt_emb = self.clip.get_text_features(**texts) # (batch_size, 512)
        
        # Encode the labels
        lbl_emb = self.label_encoder(labels) # (batch_size, 512)
        comb_emb = self.combiner(txt_emb, lbl_emb) # (batch_size, 512)
        
        return comb_emb
    
    def combine(self, txt_emb, labels):
        # Encode the labels
        lbl_emb = self.label_encoder(labels) # (batch_size, 512)
        comb_emb = self.combiner(txt_emb, lbl_emb) # (batch_size, 512)
        
        return comb_emb

    def forward(self, images, texts, labels):
        # Extract image and text features
        img_emb = self.clip.get_image_features(**images) # (batch_size, 512)
        txt_emb = self.clip.get_text_features(**texts) # (batch_size, 512)
        
        # Encode the labels
        lbl_emb = self.label_encoder(labels) # (batch_size, 512)
        
        # Combine text and label features
        comb_emb = self.combiner(txt_emb, lbl_emb) # (batch_size, 512)
        
        return img_emb, txt_emb, lbl_emb, comb_emb # For now we only need img_emb and comb_emb to calculate the loss


def main():
    # Load pretrained model
    from transformers import AutoProcessor, CLIPModel
    import requests
    from PIL import Image

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
