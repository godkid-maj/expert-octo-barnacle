# expert-octo-barnacle
Fine-tunes a ResNet18 on real photos (Oxford-IIIT Pet dataset — 37 cat/dog
breeds, downloads automatically) and serves it through a Streamlit app where
anyone can upload a photo and get live predictions.

```
image-classifier-app/
├── train.py          # fine-tunes ResNet18, saves model/model.pth + classes.json
├── app.py             # Streamlit front end
├── requirements.txt
└── model/             # created after training
    ├── model.pth
    └── classes.json
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py --epochs 8 --batch-size 32
```

What it does:
- Downloads the Oxford-IIIT Pet dataset automatically (~800 MB, one-time).
- Loads a ResNet18 pretrained on ImageNet, replaces the final layer with a
  fresh one sized for 37 classes, and trains just that layer first
  (fast, keeps the pretrained features intact).
- After `--unfreeze-after` epochs (default 4), unfreezes the whole network
  and fine-tunes everything at a lower learning rate — this is the standard
  "linear probe then fine-tune" transfer-learning recipe.
- Saves the best checkpoint (by validation accuracy) to `model/model.pth`
  and the class name list to `model/classes.json`.

Expect ~85-93% validation accuracy after 8 epochs on a GPU; CPU training
works too, just slower (~10-20 min/epoch depending on hardware).

**Using your own dataset instead:** set `USE_CUSTOM_DATA = True` at the top
of `train.py` and arrange your images as:
```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg
```
Everything else (model, training loop, app) works unchanged — it reads
class names from whatever folders it finds.

## 3. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a photo; the app shows the image
alongside the top-5 predicted classes with confidence bars. If you haven't
run `train.py` yet, the app still works — it falls back to a generic
ImageNet-pretrained ResNet18 (1000 general classes) so you can see the UI
immediately, then automatically switches to your fine-tuned model once
`model/model.pth` exists.

## How it works (for the write-up / demo)

- **Transfer learning, not training from scratch.** ResNet18 already knows
  general visual features (edges, textures, shapes) from ImageNet; we only
  need to teach it the specific classes in our dataset, which needs far
  less data and compute than training from scratch.
- **Two-phase fine-tuning.** First train only the new final layer with the
  backbone frozen (fast, stable), then unfreeze everything and continue at
  a 10x lower learning rate so the pretrained weights adjust gently instead
  of being overwritten.
- **Standard ImageNet preprocessing** (resize, center-crop to 224x224,
  normalize with ImageNet mean/std) is applied identically at train and
  inference time so the model sees consistent inputs.
- **Streamlit as the product layer.** `st.cache_resource` loads the model
  once and reuses it across requests; the rest is upload → preprocess →
  softmax → display.

## Extension ideas

- Swap in a different backbone (`models.efficientnet_b0`, `models.vit_b_16`)
  by changing `build_model()`.
- Add Grad-CAM visualization to show *why* the model predicted a class.
- Log predictions + user feedback (thumbs up/down) to build a dataset for
  active learning.
- Deploy the Streamlit app to Streamlit Community Cloud or a Docker
  container behind a simple API.
- Add a confusion matrix / per-class accuracy report after training to
  spot which breeds the model confuses most.
