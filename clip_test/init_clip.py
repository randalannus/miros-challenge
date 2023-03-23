"""Initializes the CLIP module if it has not been initialized yet."""

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip.load("ViT-B/32", device=device)
