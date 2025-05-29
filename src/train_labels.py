import pandas as pd

# Đọc file gốc
df = pd.read_csv('data/isic2018/labels/train_labels.csv')

# Lấy cột ảnh
image_ids = df['image']

# Lấy index của nhãn đúng (1) => tạo single-label
label_cols = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
df['label'] = df[label_cols].idxmax(axis=1)

# Tạo path ảnh
df['file'] = 'data/isic2018/train/' + df['image'] + '.jpg'

# Chọn cột cần thiết
df_labeled = df[['file', 'label']]

# Lưu lại file đã xử lý
df_labeled.to_csv('data/isic2018/labels/train_labeled.csv', index=False)

# Tạo file unlabeled (chỉ có file, không có nhãn)
df_unlabeled = df[['file']]
df_unlabeled.to_csv('data/isic2018/labels/train_unlabeled.csv', index=False)
