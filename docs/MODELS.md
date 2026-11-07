# Encoder providers and local weights

The study config lists all ten backbones. Feature extraction removes the original classification head; encoders are in evaluation mode with gradients disabled.

| Key | Provider | Study feature dimension |
|---|---|---:|
| `resnet50` | torchvision ResNet-50 | 2048 |
| `swin_b` | torchvision Swin-B | 1024 |
| `vit_s16` | timm `vit_small_patch16_224.augreg_in21k_ft_in1k` | 384 |
| `vit_b16` | torchvision ViT-B/16 | 768 |
| `convnext_tiny` | torchvision ConvNeXt-T | 768 |
| `convnext_base` | torchvision ConvNeXt-B | 1024 |
| `dinov3_vits16` | official DINOv3 ViT-S/16 | 384 |
| `dinov3_vitb16` | official DINOv3 ViT-B/16 | 768 |
| `dinov3_convnext_tiny` | official DINOv3 ConvNeXt-T | 768 |
| `dinov3_convnext_base` | official DINOv3 ConvNeXt-B | 1024 |

Local checkpoint filenames in `study.yaml` identify the pretrained variants actually used; a different pretrained variant is a new experiment, not a numerically exact reproduction. The local ResNet file is `resnet50-0676ba61.pth`; do not silently substitute a different “DEFAULT” weight release. Local state dictionaries are loaded strictly. Models without local weights raise an error unless `allow_download: true` is explicitly set for torchvision/timm. Downloads then follow upstream providers and must be reported.

Obtain DINOv3 code and licensed weights from the [official repository](https://github.com/facebookresearch/dinov3). Put/clone that code at the configured `repository`, and point `weights` to the file obtained under its terms. The wrapper imports only the official `dinov3.hub.backbones` module, avoiding optional segmentation dependencies imported by the full hub configuration. DINOv3 sources/weights are not vendored in this project. Record your actual upstream commit; this implementation additionally fingerprints architecture source files.

Providers return `pooled: [B,D]`. Dense DINOv3 ViTs also return `cls: [B,depth,D]` and `tokens: [B,L,P+1,D]` for configured one-based layer numbers, with token 0 the CLS token. Register tokens are handled by the upstream intermediate-layer API. At 224×224 with 16-pixel patches, P=196. Custom encoders may return another dimension/depth; head dimensions are inferred from caches. Region explanations currently require a square patch grid, though the image source itself may have arbitrary original dimensions.

A custom model factory `module:function` receives the model specification and returns a frozen feature module whose forward returns the above dictionary. Configure `provider: custom` and `factory`. A changed custom implementation should use a fresh output/cache directory; add its implementation identity to your model settings. The provided toy provider is for tests only.

References: [torchvision model API](https://docs.pytorch.org/vision/main/models.html), [timm feature API](https://huggingface.co/docs/timm/feature_extraction), [DINOv3 upstream](https://github.com/facebookresearch/dinov3).
