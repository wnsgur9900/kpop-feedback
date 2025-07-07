# train.py (개선판)

import torch
from torch.utils.data import DataLoader, random_split
from dataset import Pose3DDataset
from model import Pose3DNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 하이퍼파라미터
root_dir = "data-files"
num_joints = 29  # 관절 수
batch_size = 64
epochs = 30
lr = 1e-4
val_ratio = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # 전체 데이터셋 로드
    full_dataset = Pose3DDataset(root_dir)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델, 손실, 최적화
    model = Pose3DNet(num_joints).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (1) 그래프를 위한 로그 리스트들
    train_loss_log = []
    val_loss_log = []
    rmse_log = []
    
    best_val_loss = float('inf')

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, poses in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training"):
            imgs, poses = imgs.to(device), poses.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, poses)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)



        # Validation 평가
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, poses in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation"):
                imgs, poses = imgs.to(device), poses.to(device)
                preds = model(imgs)
                loss = criterion(preds, poses)
                val_loss += loss.item()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(poses.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # MSE, RMSE, (선택) Cosine Similarity 등 계산
        y_pred = np.concatenate(all_preds, axis=0).reshape(-1, num_joints*3)
        y_true = np.concatenate(all_labels, axis=0).reshape(-1, num_joints*3)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        train_loss_log.append(avg_train_loss)
        val_loss_log.append(avg_val_loss)
        rmse_log.append(rmse)

        print(f"🔁 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | RMSE: {rmse:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_pose3d_model.pth")
            print(f"💾 모델 저장됨 (val_loss={avg_val_loss:.6f})")

    plt.figure(figsize=(10,6))
    plt.plot(train_loss_log, label='Train Loss')
    plt.plot(val_loss_log, label='Val Loss')
    plt.plot(rmse_log, label='RMSE')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")  # 이미지 저장 (선택)
    plt.show()
