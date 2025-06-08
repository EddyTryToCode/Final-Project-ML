import pandas as pd
import os

# Đường dẫn đến thư mục labels
label_dir = '/teamspace/studios/this_studio/Final-Project-ML/data/isic2018/labels'

# 1. Train
import pandas as pd

# Đọc file one-hot
df = pd.read_csv('/teamspace/studios/this_studio/Final-Project-ML/data/isic2018/labels/train_labels.csv')

# Xác định các cột nhãn
label_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Tìm chỉ số của nhãn (label_idx)
df['label_idx'] = df[label_columns].values.argmax(axis=1)

# Tìm tên của nhãn (label)
df['label'] = df[label_columns].idxmax(axis=1)

# Chỉ lấy các cột cần thiết
df_out = df[['image', 'label', 'label_idx']]
df_out.to_csv('data/isic2018/labels/train_labeled_idx.csv', index=False)

print("✅ Đã tạo xong file train_labeled_idx.csv ")



# 2. Validation
val_path = os.path.join(label_dir, 'val_labels.csv')

# Đọc file val
df_val = pd.read_csv(val_path)

# Danh sách nhãn
label_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Tìm index (vị trí) của nhãn == 1
df_val['label'] = df_val[label_cols].idxmax(axis=1)

# Giảm thành 2 cột: image + label
df_val = df_val[['image', 'label']]

# Lưu lại
out_path = os.path.join(label_dir, 'val_idx.csv')
df_val.to_csv(out_path, index=False)
# Map label string -> index
label2idx = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
df_val['label_idx'] = df_val['label'].map(label2idx)

# Lưu file có index
df_val.to_csv(os.path.join(label_dir, 'val_idx.csv'), index=False)
print("✅ Saved val_idx_num.csv with label index.")

TEST_DIR  = '../data/isic2018/test'
label_cols = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']

df_test = pd.read_csv(os.path.join(label_dir, 'test_labels.csv'))
# chuyển one-hot → label, label_idx
df_test['label']     = df_test[label_cols].idxmax(axis=1)
df_test['label_idx'] = df_test[label_cols].values.argmax(axis=1)
df_test[['image','label','label_idx']].to_csv(
    os.path.join(label_dir, 'test_idx.csv'), index=False
)
print(f"Đã tạo test_idx.csv với {len(df_test)} dòng.")
