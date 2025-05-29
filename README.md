# Self-Supervised SimCLR Pretraining for Skin Lesion Classification

## Overview
This repository implements a two-stage pipeline on the ISIC 2018 Task 3 dataset:
1. **Self-Supervised Pretraining** with SimCLR on 10 015 unlabeled dermoscopic images  
2. **Full-Model Fine-Tuning** on the same 10 015 labeled images, evaluated on 1 000 hold-out images  

We achieve **78.24 %** balanced accuracy on the official validation set using ResNet-18.

## Directory Structure

```plaintext
final-project-ml/
├── data/
│   └── isic2018/
│       ├── train/                # (ignored by git) raw train images
│       ├── val/                  # (ignored by git) raw val images
│       └── labels/               # tracked CSVs only
│           ├── train_unlabeled.csv
│           ├── train_labeled_idx.csv
│           ├── val_idx.csv
│           ├── val_labels.csv 
│           ├── train_labels.csv
│           └── train_labeled.csv
│
├── figures/                      # tracked: plots for report
│   ├── loss_curve.png
│   ├── acc_curve.png
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── notebooks/                    # tracked
│   ├── Pretraining-SimCLR.ipynb
│   └── FineTune-SimCLR.ipynb
│
├── src/                          # tracked
│   ├── create_idx_csv.py
│   ├── datasets.py
│   ├── simclr_model.py
│   └── finetune_model.py
│
├── scripts/                      # tracked (optional CLI wrappers)
│   ├── train_simclr.py
│   └── train_finetune.py
│
├── checkpoints/                  # tracked (optional small ckpt files)
│   ├── simclr_encoder.pth
│   └── best_finetuned.pth
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/                  # copy of plots for LaTeX
│
├── environment.yml               # tracked
├── requirements.txt              # tracked
├── README.md                     # this file
└── .gitignore                    # see below

Getting Started
1. Clone & create environment
2. Download ISIC 2018 data
3. Preprocess labels
4. Pretrain SimCLR
5. Fine-tune & evaluate
6. Build report

