# CLIP Paper Note

**Paper:** Learning Transferable Visual Moels From Natural Language Supervision

**Authors:** Radford et al. (OpenAI, 2021)



## Summary

CLIP learns to align images and texts in a shared high-dimensional space through contrastive learning, enabling zero-shot classification on diverse datasets like ImageNet.

## Architecture

**Image Encoder:** ResNet or ViT

**Text Encoder:** Decoder-only Transformer (GPT-2 style)

Both encoders map their inputs to a shared high-dimensional space through linear projection heads.

**Loss Function:** Symmetric Cross-Entropy

Logits are computed via cosine similarity and scaled by a temperature parameter. Then, the similarity matrix is employed to compute the Image-to-Text Cross-Entropy and Text-to-Image Cross-Entropy. The final Symmetric Cross-Entropy is computed by averaging the two entropy losses.



![](D:\marktextimage\2026-03-10-21-43-56-image.png)

## Training

**Dataset:** WIT (WebImageText), 400M image-text pairs collected from the Internet

**Loss Function:** Symmetric cross-entropy

For a batch of N image-text pairs, there are


