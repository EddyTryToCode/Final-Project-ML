```markdown
# Skin Lesion Classification with SimCLR + FixMatch++

This repository implements a state-of-the-art semi-supervised pipeline on the ISIC 2018 Task 3 dataset (skin lesion classification) by combining:

1. **SimCLR** self-supervised pretraining on ~9 000 unlabeled images  
2. **FixMatch++** fine-tuning on 10% labeled (~1 000) + unlabeled images with:
   - Dynamic pseudo-label threshold
   - MixUp on both labeled & pseudo batches
   - Label smoothing
   - EMA teacher
3. **OneCycleLR** scheduling, **top-3 checkpoint** ensembling, and **Test-Time Augmentation (TTA)**  

---

## 📂 Repository Structure

```

.
├── data/
│   └── isic2018/
│       ├── labels/
│       │   ├── train\_labeled.csv      # \~1 001 labeled
│       │   ├── train\_unlabeled.csv    # \~9 014 unlabeled
│       │   ├── val_idx.csv                # 193 validation
│       │   └── test\_idx.csv           # 1 512 test
│       ├── train/                     # all train images (.jpg)
│       ├── val/                       # validation images
│       └── test/                      # test images
├── notebooks/
│       ├── checkpoints (fix match path after semi-supervised)
│       ├── Pretrained\_SimCLR\_Model.ipynb
│       └── FineTune-SimCLR(latest).ipynb
├── src/
│   ├── create\_idx\_csv.py
│   ├── create\_splits.py
│   └── train\_labels.py 
├── paper/
│   └── ML\_Final.pdf
├── environment.yml
└── README.md

````

---

## 🔧 Environment & Installation

1. **Clone** this repo and enter directory:
   ```bash
   git clone <https://github.com/EddyTryToCode/Final-Project-ML>
````

2. **Create conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate <env_name>
   ```

   Or use `pip install -r requirements.txt` if provided.

---

## 📊 Data Preparation

Download ISIC2018 Task 3 

Place images under `data/isic2018/`:

* **Images**:

  * `data/isic2018/train/`
  * `data/isic2018/val/`
  * `data/isic2018/test/`

use all "tool" in src to change orginal labels

* **Labels CSVs** (`data/isic2018/labels/`):

  * `train_labeled.csv` (10% \~1 001)
  * `train_unlabeled.csv` (\~9 014)
  * `val.csv` (193)
  * `test_idx.csv` (1 512)

CSV format:

```csv
image,label,label_idx
ISIC_0000000,MV,1
ISIC_0000001,NV,0
...
```
---

## 🚀 Usage

### 1. Pretrain SimCLR

- Use notebook : .../notebooks/Pretrained_SimCLR_ Model.ipynb

### 2. Fine-tune with FixMatch++ + Ensemble & TTA on Test

- Use notebook : .../notebooks/FineTune-SimCLR(lastest).ipynb


## 📈 Results

| Method                     |   Val ACC  |  Test ACC  |
| -------------------------- | :--------: | :--------: |
| Supervised (ResNet18)      |   54–60%   |   59–64%   |
| FixMatch (baseline)        |   60–65%   |   63–66%   |
| **This work (FixMatch++)** | **73–75%** | **70.30%** |

---

## 📚 Citation

```bibtex
@article{codella2019skin,
  title={Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}

@inproceedings{yourname2025fixmatchpp,
  title={FixMatch++: Enhanced Semi-Supervised Learning for Skin Lesion Classification},
  author={YourName, First and Coauthor, Second},
  booktitle={NeurIPS Workshop on Medical AI},
  year={2025}
}

@article{codella2019skin,
  title={Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and Celebi, Emre and Dusza, Stephen and Gutman, David and Helba, Brian and Kalloo, Aadi and Liopyris, Konstantinos and Marchetti, Michael and Kittler, Harald and Halpern, Allan},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019},
  url={https://arxiv.org/abs/1902.03368}
}
```


