# visualize.py

import torch
from model import Pose3DNet
from dataset import Pose3DDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 설정
num_joints = 29
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = Pose3DNet(num_joints).to(device)
model.load_state_dict(torch.load("best_pose3d_model.pth"))
model.eval()

# 테스트 데이터 1개 가져오기
dataset = Pose3DDataset("data-files/test")
img, gt_pose = dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', label='Predicted')
ax.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], c='b', alpha=0.5, label='Ground Truth')
ax.legend()
plt.title("3D Pose Prediction vs Ground Truth")
plt.show()
