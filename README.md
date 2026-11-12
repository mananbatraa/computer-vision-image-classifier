# AI Computer Vision Image Classifier

Upload a satellite image and the model says what kind of land it shows (forest, river, highway, residential…), with a confidence score for each class.

It's a MobileNetV3 network pretrained on ImageNet and fine-tuned with transfer learning on **EuroSAT**: 27,000 Sentinel-2 satellite images, 64×64 pixels, in 10 land-use classes. ImageNet doesn't contain these classes, so the network genuinely has to adapt rather than reuse categories it already knows.

Python · PyTorch · torchvision · MobileNetV3 · scikit-learn · Streamlit

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
python -m classifier.train              # downloads EuroSAT (~90 MB) on first run; ~6 min on an Apple M-series GPU
python -m classifier.predict examples/River.jpg --top 3
streamlit run app.py
pytest
```

Quick run on a small subset: `python -m classifier.train --limit-per-class 300 --epochs-head 1 --epochs-finetune 1`.

## How it works

| Stage | File | Details |
|---|---|---|
| Dataset preparation | `classifier/data.py` | Stratified 70/15/15 train/validation/test split (18,900 / 4,050 / 4,050 images), so every class has the same share in each split. |
| Preprocessing | `classifier/data.py` | Resize to 128×128 and normalise with ImageNet statistics. Training adds flips, 90° rotations and colour jitter; a satellite tile is still valid at any orientation. |
| Model | `classifier/model.py` | MobileNetV3-Large with its 1000-class ImageNet head replaced by a 10-class layer. |
| Transfer learning | `classifier/train.py` | **Phase 1:** freeze the pretrained feature extractor and train only the new head (1.2M parameters). **Phase 2:** unfreeze the last 4 blocks and fine-tune at a lower learning rate (3.4M parameters). AdamW, cosine schedule, label smoothing. The best validation epoch is kept. |
| Prediction | `classifier/predict.py` | Softmax probabilities, top-k classes with confidence, and a low-confidence flag below 50% (useful when the input looks nothing like satellite imagery). |
| App | `app.py` | Streamlit page: upload an image, or pick one of 10 held-out test images in `examples/`, and see the top-5 classes with confidence bars. |

Runs on CUDA, Apple Silicon (MPS) or CPU, chosen automatically.

## Results

Held-out test set: 4,050 images never used for training or model selection.

| Metric | Value |
|---|---|
| Test accuracy | **97.4%** |
| Macro F1 | **0.973** |
| Training time | 6.1 min (Apple M-series GPU) |

Validation accuracy by epoch: 89.7% and 92.0% with only the head trained, then 96.1%, 96.8%, 97.3% and 97.3% once the last blocks were fine-tuned. Unfreezing part of the network clearly helps.

| Class | F1 | Class | F1 |
|---|---|---|---|
| SeaLake | 0.993 | River | 0.969 |
| Forest | 0.989 | Highway | 0.967 |
| Residential | 0.989 | AnnualCrop | 0.960 |
| Industrial | 0.985 | PermanentCrop | 0.956 |
| HerbaceousVegetation | 0.971 | Pasture | 0.953 |

**Where it goes wrong:** the most common mistakes are PermanentCrop ↔ AnnualCrop, Highway → River (both are thin lines across a tile), and Pasture → HerbaceousVegetation. These are visually close classes even for people. The `examples/Highway.jpg` demo image is one of these mistakes: the model says Pasture at 73% and Highway at 23%. Full per-class metrics and the confusion matrix are in [`reports/metrics.json`](reports/metrics.json).

## Limitations
- Trained only on 64×64 RGB Sentinel-2 tiles from Europe. Ordinary photos (a cat, a street) get a forced choice among these 10 classes; the low-confidence flag helps, but it isn't an out-of-distribution detector.
- Uses only the RGB bands, not EuroSAT's 13 multispectral bands, which would likely help separate the crop types.
- Next steps: compare against a from-scratch CNN baseline, calibrate the confidence scores, add Grad-CAM heatmaps to show which pixels drove a prediction.
