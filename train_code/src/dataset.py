# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

class Pose3DDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_paths = []
        self.labels = []
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        # 모든 data_x 폴더 순회
        for folder in sorted(self.root_dir.glob("data_*")):
            images_dir = folder / "images"
            features_path = folder / "features.npy"

            if not features_path.exists():
                continue

            # 프레임별 이미지와 해당 feature 일치하는지 확인
            img_files = sorted(images_dir.glob("*.jpg"))
            features = np.load(features_path)

            if len(img_files) != len(features):
                print(f"{folder.name}: 이미지 {len(img_files)}장 ≠ 피처 {len(features)}개")


            min_len = min(len(img_files), len(features))
            if min_len < 10:
                print(f"⚠️ 프레임 부족 → 건너뜀: {folder.name}")
                continue

            self.image_paths += img_files[:min_len]
            self.labels.append(features[:min_len])

        self.labels = np.concatenate(self.labels, axis=0)  # 전체 (N_total, J, 3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        pose3d = self.labels[idx]

        # pose3d가 1차원 벡터라면 reshape 해주기
        if pose3d.ndim == 1:
            pose3d = pose3d.reshape(-1, 3)

        pose3d = torch.tensor(pose3d, dtype=torch.float32)

        return image, pose3d

