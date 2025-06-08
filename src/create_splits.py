# src/create_splits.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Thư mục chứa CSV labels
    LABEL_DIR = 'data/isic2018/labels'
    # Đường dẫn tới thư mục ảnh train/val (dùng ../ nếu notebook/script nằm cấp trên)
    TRAIN_DIR = '../data/isic2018/train'
    VAL_DIR   = '../data/isic2018/val'

    os.makedirs(LABEL_DIR, exist_ok=True)

    # --- 1) Xử lý train set ---
    # Đọc CSV gốc có one-hot labels
    df_train = pd.read_csv(os.path.join(LABEL_DIR, 'train_labels.csv'))

    # Các cột one-hot
    label_cols = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']

    # Tạo cột label (tên) và label_idx (số)
    df_train['label']     = df_train[label_cols].idxmax(axis=1)
    df_train['label_idx'] = df_train[label_cols].values.argmax(axis=1)

    # Tách 90% làm unlabeled, 10% còn lại labeled
    df_labeled, df_unlabeled = train_test_split(
        df_train,
        test_size=0.90,
        stratify=df_train['label'],
        random_state=42
    )

    # 1.1 Tạo train_unlabeled.csv
    unlabeled_paths = [f"{TRAIN_DIR}/{img_id}.jpg" for img_id in df_unlabeled['image']]
    df_unl = pd.DataFrame({'file': unlabeled_paths})
    df_unl.to_csv(os.path.join(LABEL_DIR, 'train_unlabeled.csv'), index=False)
    print(f"Đã tạo train_unlabeled.csv với {len(df_unl)} dòng.")

    # 1.2 Tạo train_labeled_idx.csv
    df_lab = df_labeled[['image','label','label_idx']].copy()
    df_lab.to_csv(os.path.join(LABEL_DIR, 'train_labeled_idx.csv'), index=False)
    print(f"Đã tạo train_labeled_idx.csv với {len(df_lab)} dòng.")

    # --- 2) Xử lý validation set riêng ---
    df_val = pd.read_csv(os.path.join(LABEL_DIR, 'val_labels.csv'))  # one-hot
    # Tạo label & label_idx giống trên
    df_val['label']     = df_val[label_cols].idxmax(axis=1)
    df_val['label_idx'] = df_val[label_cols].values.argmax(axis=1)

    # Tạo val_idx.csv
    df_val_out = df_val[['image','label','label_idx']].copy()
    df_val_out.to_csv(os.path.join(LABEL_DIR, 'val_idx.csv'), index=False)
    print(f"Đã tạo val_idx.csv với {len(df_val_out)} dòng.")

if __name__ == '__main__':
    main()
