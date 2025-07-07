import os
import json
import numpy as np
from tqdm import tqdm

# 루트 폴더
root_dir = "data-files/test"

# data_1 ~ data_99 순회
for i in range(100, 111):
    folder_name = f"data_{i}"
    label_dir = os.path.join(root_dir, folder_name, "labels")
    output_path = os.path.join(root_dir, folder_name, "features.npy")
    
    if not os.path.isdir(label_dir):
        print(f"❌ 라벨 폴더 없음 → 건너뜀: {folder_name}")
        continue

    features = []
    filenames = sorted(os.listdir(label_dir))
    
    for file in filenames:
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(label_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ann = data.get("annotations", [])
        if not ann:
            print(f"⚠️ annotations 비어있음 → {file}")
            continue
        
        keypoints3d = ann[0].get("keypoints3d", [])
        if len(keypoints3d) != 87:
            print(f"⚠️ keypoints3d 길이 오류({len(keypoints3d)}) → {file}")
            continue

        keypoints3d_np = np.array(keypoints3d).reshape(-1, 3)
        features.append(keypoints3d_np)

    if not features:
        print(f"⚠️ 유효한 feature 없음 → {folder_name}")
        continue

    features_array = np.array(features)  # shape: (N, 29, 3)
    np.save(output_path, features_array)
    print(f"✅ 저장 완료 → {output_path} ({features_array.shape})")
