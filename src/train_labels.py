import pandas as pd

# Đọc file gốc
df = pd.read_csv('data/isic2018/labels/train_unlabeled.csv')

# Lấy cột ảnh
image_ids = df['image']

# Tạo path ảnh
df['file'] = 'data/isic2018/train/' + df['image'] + '.jpg'

# Tạo file unlabeled (chỉ có file, không có nhãn)
df_unlabeled = df[['file']]
df_unlabeled.to_csv('data/isic2018/labels/train_unlabeled.csv', index=False)
