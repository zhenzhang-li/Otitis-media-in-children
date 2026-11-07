# Third-party dependencies and model materials

The repository contains project utilities and a refactored implementation of the study's fusion/analysis pipeline. It does **not** bundle upstream DINOv3 source, pretrained model weights, clinical images or clinical metadata. The project MIT license does not relicense those external materials.

- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision): imported installed packages; consult their upstream license files and model documentation.
- [timm](https://github.com/huggingface/pytorch-image-models): installed provider for the supervised ViT-S baseline; pretrained sources/variants have their own provenance.
- [DINOv3](https://github.com/facebookresearch/dinov3): official implementation and weights are subject to the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Readers obtain the code/weights separately under those terms. The manuscript and subsequent work should acknowledge the model as required by its upstream terms.
- NumPy, pandas, SciPy, scikit-learn, Pillow, PyYAML, matplotlib and openpyxl are installed dependencies. Their upstream licensing remains applicable.

The listed sources document APIs and provenance; this notice does not grant access to restricted pretrained files or change their conditions. Do not copy private data, gated download URLs or access tokens into the repository.
