import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- 数据集封装 ----------------
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- 预训练 ResNet 模型 ----------------
class ResNetRegressor(nn.Module):
    def __init__(self, output_dim=3):
        super(ResNetRegressor, self).__init__()
        # 使用 ResNet18 架构，不从互联网下载预训练权重
        self.resnet = models.resnet18(weights=None)

        # 加载本地预训练权重
        state_dict = torch.load(r"F:\\桌面\\文件\\学习\\竞赛\\国创赛\\光谱\\predict\\project\\resnet18-f37072fd.pth")
        self.resnet.load_state_dict(state_dict)

        # 冻结所有层，除了最后一层
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 修改最后的全连接层来适应回归任务（3个输出）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        return self.resnet(x)

# ---------------- 数据处理函数 ----------------
def load_data_from_excels(folder):
    spectra, targets = [], []
    for file in Path(folder).glob("[0-9][0-9][0-9].xlsx"):
        try:
            df = pd.read_excel(file, header=None, engine='openpyxl')
            if df.shape[0] < 4 or df.shape[1] < 18:
                continue

            spec = df.iloc[:3, :18].replace('[^0-9.]', '', regex=True).astype(float)
            quality = df.iloc[3, :3].replace('[^0-9.]', '', regex=True).astype(float)
            if quality.isnull().any():
                continue

            q = quality.values.copy()
            q[0] *= 100  # 水分

            for i in range(3):
                spectra.append(spec.iloc[i].values)
                targets.append(q)
        except Exception as e:
            continue
    return np.array(spectra), np.array(targets)

# ---------------- 主训练流程 ----------------
def train_model(X, y, epochs=100, batch_size=32, lr=1e-3):
    scaler_x = StandardScaler()
    X_sg = savgol_filter(X, 5, 2, deriv=1, axis=1)
    X_scaled = scaler_x.fit_transform(X_sg)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    train_loader = DataLoader(SpectraDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SpectraDataset(X_val, y_val), batch_size=batch_size)

    model = ResNetRegressor(output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.cat([model(xb) for xb, _ in val_loader])
            val_true = torch.cat([yb for _, yb in val_loader])
            val_loss = mean_squared_error(val_true.numpy(), val_preds.numpy())
            val_r2 = r2_score(val_true.numpy(), val_preds.numpy())
            print(f"Epoch {epoch+1:03d} | Val Loss: {val_loss:.4f} | R²: {val_r2:.3f}")

    return model, scaler_x, scaler_y

# ---------------- 预测函数 ----------------
def predict(model, X_raw, scaler_x, scaler_y):
    X_sg = savgol_filter(X_raw, 5, 2, deriv=1, axis=1)
    X_scaled = scaler_x.transform(X_sg)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        preds = model(X_tensor).numpy()
    return scaler_y.inverse_transform(preds)

# ---------------- 运行主流程 ----------------
if __name__ == "__main__":
    data_dir = r"F:\\桌面\\文件\\学习\\竞赛\\国创赛\\光谱\\predict\\project\\data"
    X, y = load_data_from_excels(data_dir)
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

    model, scaler_x, scaler_y = train_model(X, y)

    test_sample = X[np.random.randint(0, len(X))].reshape(1, -1)
    prediction = predict(model, test_sample, scaler_x, scaler_y)
    print("\n预测结果：")
    print(f"水分: {prediction[0, 0]:.1f}%")
    print(f"硬度: {prediction[0, 1]:.2f} N/cm²")
    print(f"糖度: {prediction[0, 2]:.2f} °Brix")
