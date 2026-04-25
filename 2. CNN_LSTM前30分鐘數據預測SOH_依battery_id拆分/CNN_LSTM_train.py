import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 數據集定義
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
# 2. CNN-LSTM 混合模型架構
# ==========================================
class BatteryCNNLSTM(nn.Module):
    def __init__(self, input_channels=4, hidden_size=64):
        super(BatteryCNNLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

# ==========================================
# 3. 訓練與完整視覺化
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    train_ds = BatteryDataset('X_train.npy', 'y_train.npy')
    val_ds = BatteryDataset('X_val.npy', 'y_val.npy')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = BatteryCNNLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20000
    patience = 200
    best_val_loss = float('inf')
    counter = 0
    
    train_losses = []
    val_losses = []

    print("開始訓練 CNN-LSTM 模型...")
    for epoch in range(epochs):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_CNN_LSTM.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 4. 繪製多圖視覺化
    if os.path.exists('best_model_CNN_LSTM.pth'):
        model.load_state_dict(torch.load('best_model_CNN_LSTM.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    plt.figure(figsize=(15, 10))

    # 圖 1: Loss 曲線
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('CNN-LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    # 圖 2: 預測 vs 真實
    plt.subplot(2, 2, 2)
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.title('Actual vs Predicted Capacity (CNN-LSTM)')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Predicted Capacity (Ah)')

    # 圖 3: 誤差分佈
    plt.subplot(2, 2, 3)
    errors = all_preds - all_targets
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Error Distribution (CNN-LSTM)')
    plt.xlabel('Error (Ah)')
    plt.ylabel('Frequency')

    # 圖 4: SOH 誤差絕對值
    plt.subplot(2, 2, 4)
    errors_soh = (all_preds - all_targets) / 2.0 * 100
    abs_errors_soh = np.abs(errors_soh)
    mae_soh = np.mean(abs_errors_soh)
    plt.hist(abs_errors_soh, bins=30, color='green', edgecolor='black', alpha=0.6)
    plt.axvline(x=mae_soh, color='red', linestyle='--', label=f'SOH MAE = {mae_soh:.4f}')
    plt.title('SOH Error Magnitude Distribution')
    plt.xlabel('|Predicted - Actual| SOH')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results_CNN_LSTM.png')
    print("\n訓練完成！多圖診斷報告已儲存為 'training_results_CNN_LSTM.png'")

if __name__ == "__main__":
    main()
