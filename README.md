````markdown
# Deep Image Classification with PyTorch

A modular PyTorch project demonstrating:

- Training a ResNet‐style convolutional neural network (CNN) from scratch  
- Transfer learning with pretrained torchvision models  
- Data loading with augmentation, validation splits, and test sets  
- Custom training loop with learning‐rate scheduling and checkpointing  
- A scripted `Predictor` wrapper for easy inference  
- Interactive HTML demos of training and transfer‐learning workflows

---

## 🚀 Features

- **CNN from Scratch**: Build and train a residual‐block network (`src/model.py`)  
- **Transfer Learning**: Swap in any `torchvision` model and fine‐tune its final layer (`src/transfer.py`)  
- **Data Pipeline**: Automatic mean/std computation, augmentation, train/val/test splits (`src/data.py`)  
- **Training Utilities**:  
  - `train_one_epoch` & `valid_one_epoch` loops with GPU support  
  - Cosine‐annealing LR scheduler & checkpointing (`train.py`)  
  - Configurable loss (cross‐entropy with smoothing) & optimizers (SGD/AdamW) (`src/optimization.py`)  
- **Inference API**: `Predictor` class for resizing, centering, normalizing and softmax output (`src/predictor.py`)  
- **Live Loss Plotting**: Optional interactive plots via `livelossplot`  
- **Demos**: Prebuilt HTML outputs of notebook workflows:  
  - `app.html` — end‐to‐end training demo  
  - `cnn_from_scratch.html` — building a CNN by hand  
  - `transfer_learning.html` — fine‐tuning pretrained models  

---

## 📦 Requirements

- Python 3.8+  
- PyTorch & torchvision  
- NumPy  
- Matplotlib  
- tqdm  
- livelossplot  

Install with:

```bash
pip install torch torchvision numpy matplotlib tqdm livelossplot
````

---

## 📁 Project Structure

```
.
├── train.py                  # Main training & validation loop, checkpointing
├── src/
│   ├── data.py               # Data loaders, transforms, train/val/test splits
│   ├── model.py              # ResidualBlock & MyModel CNN definition
│   ├── optimization.py       # get_loss(), get_optimizer() utilities
│   ├── predictor.py          # Predictor wrapper & test harness
│   ├── transfer.py           # get_model_transfer_learning() helper
│   └── helpers.py            # Data‐path & mean/std computation utilities
├── app.html                  # Embedded demo of full training pipeline
├── cnn_from_scratch.html     # Notebook output: CNN built from scratch
├── transfer_learning.html    # Notebook output: transfer‐learning workflow
└── README.md                 # (this file)
```

---

## 🔧 Usage

### 1. Prepare your data

Organize your images in:

```
dataset_root/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── …
└── test/
    ├── class_0/
    ├── class_1/
    └── …
```

Set the `DATA_DIR` environment variable or modify `src/helpers.py:get_data_location()`.

---

### 2. Training from Scratch

```bash
python train.py \
  --data-dir /path/to/dataset_root \
  --batch-size 32 \
  --epochs 30 \
  --lr 0.01 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --save-path checkpoints/ \
  --interactive
```

Key behaviors:

* Uses `get_data_loaders()` for train/val/test (`src/data.py`)
* Builds `MyModel` from `src/model.py`
* Uses cross‐entropy with label smoothing and SGD/AdamW (`src/optimization.py`)
* Cosine‐annealing LR scheduler, saves when val loss drops >1%

---

### 3. Transfer Learning

In your Python script or notebook:

```python
from src.transfer import get_model_transfer_learning
from src.data import get_data_loaders
from src.optimization import get_loss, get_optimizer

# Load pretrained ResNet18, freeze backbone
model = get_model_transfer_learning(model_name="resnet18", n_classes=50)

loaders = get_data_loaders(batch_size=32, valid_size=0.2)
criterion = get_loss()
optimizer = get_optimizer(model, optimizer="adam", learning_rate=1e-3)

# Then call train.optimize(...) as in train.py
```

---

### 4. Inference with Predictor

```python
import torch
from src.predictor import Predictor
from src.model import MyModel
from src.helpers import compute_mean_and_std

# Instantiate your trained model
model = MyModel(num_classes=50)
model.load_state_dict(torch.load("best_model.pt"))

mean, std = compute_mean_and_std()
class_names = ["cat", "dog", …]

predictor = Predictor(model, class_names, mean, std)

# Predict on a batch of images
imgs = torch.randn(4, 3, 224, 224)  # e.g., from torchvision.transforms.ToTensor()
probs = predictor(imgs)             # shape: [4, 50], rows sum to 1
```

---

## 💡 Interactive Demos

Open the HTML files in a browser to explore ready‐made notebook outputs:

* **app.html** — Full end‐to‐end demo
* **cnn\_from\_scratch.html** — Building your own CNN
* **transfer\_learning.html** — Fine‐tuning pretrained models

---

## 🤝 Contributing

This project was developed as part of the **Udacity Nanodegree** program. Contributions and feedback from fellow Udacity Nanodegree students are welcome!

---

## 📜 License

MIT License © Nour Shammaa

