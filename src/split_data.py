import os
import shutil
import random

random.seed(42)  
data_dir = 'data/raw'
split_dir = 'data/split'
classes = ['cat', 'fruit', 'monkey']
splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for cls in classes:
    os.makedirs(f'{split_dir}/train/{cls}', exist_ok=True)
    os.makedirs(f'{split_dir}/val/{cls}', exist_ok=True)
    os.makedirs(f'{split_dir}/test/{cls}', exist_ok=True)
    images = os.listdir(f'{data_dir}/{cls}')
    random.shuffle(images)
    n = len(images)
    n_train, n_val = int(n * splits['train']), int(n * splits['val'])
    for img in images[:n_train]:
        shutil.copy(f'{data_dir}/{cls}/{img}', f'{split_dir}/train/{cls}/{img}')
    for img in images[n_train:n_train + n_val]:
        shutil.copy(f'{data_dir}/{cls}/{img}', f'{split_dir}/val/{cls}/{img}')
    for img in images[n_train + n_val:]:
        shutil.copy(f'{data_dir}/{cls}/{img}', f'{split_dir}/test/{cls}/{img}')