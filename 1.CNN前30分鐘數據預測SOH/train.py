import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 數據集定義 (Dataset)
# ==========================================
class BatteryDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

# ==========================================
# 2. 1D CNN 模型架構
# ==========================================
class BatteryCNN(nn.Module):
    def __init__(self):
        super(BatteryCNN, self).__init__()
        # 輸入形狀: (Batch, 3, 61)
        self.conv_layers = nn.Sequential(
            # 第一層卷積: 捕捉局部電壓/溫度變化
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 長度 61 -> 30
            
            # 第二層卷積: 提取更高階特徵
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 全域平均池化: 將時間維度壓縮為 1，保留特徵強處
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # 輸出預測容量
        )
        
    def forward(self, x):
        x = self.conv_layers(x) # (Batch, 64, 1)
        x = x.view(x.size(0), -1) # (Batch, 64)
        x = self.fc_layers(x)
        return x

# ==========================================
# 3. 訓練與視覺化主程式
# ==========================================
def main():
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 載入數據
    train_ds = BatteryDataset('X_train.npy', 'y_train.npy')
    val_ds = BatteryDataset('X_val.npy', 'y_val.npy')
    test_ds = BatteryDataset('X_test.npy', 'y_test.npy')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 初始化模型、損失函數與優化器
    model = BatteryCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練參數
    epochs = 20000
    patience = 150 # 早停步數
    best_val_loss = float('inf')
    counter = 0
    
    train_losses = []
    val_losses = []

    print("開始訓練...")
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 驗證階段
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 早停邏輯
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ==========================================
    # 4. 評估與繪圖
    # ==========================================
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # 繪製圖表
    plt.figure(figsize=(18, 5))

    # 圖 1: Loss 曲線
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    # 圖 2: 預測 vs 真實 (Regression Plot)
    plt.subplot(1, 3, 2)
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.title('Actual vs Predicted Capacity')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Predicted Capacity (Ah)')

    # 圖 3: 誤差分佈 (Error Histogram)
    plt.subplot(1, 3, 3)
    errors = all_preds - all_targets
    plt.hist(errors, bins=30, color='green', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Ah)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("\n訓練完成！結果圖表已儲存為 'training_results.png'")
    print(f"測試集平均絕對誤差 (MAE): {np.mean(np.abs(errors)):.4f} Ah")

if __name__ == "__main__":
    main()
