# DNN feature extraction

This folder contains the standalone pipeline used to extract the DNN features
referenced in the manuscript.

Models:

- AlexNet
- VGG-16
- ResNet-50
- CLIP-ViT-B/16
- CLIP-RN50
- DINOv2-ViT-B/14
- DINOv3-ViT-B/16

Images are resized to a shortest edge of 256 px and center-cropped to 224 × 224.
Convolutional activations are reduced by global average pooling. Intermediate
transformer activations use mean patch tokens; final outputs use the CLS token.

Usage:

```bash
conda env create -f analysis/dnn/environment.yml
conda activate primate_align-dnn
python analysis/dnn/extract_features.py
```

Input and output paths are set in `layer_config.toml`. `discover_layers.py`,
`layer_gallery.csv`, and `arch/` are included as layer-name references.
