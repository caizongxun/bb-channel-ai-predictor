"""
BB通道AI预测模型 GPU训练脚本
使用LSTM + PyTorch + GPU划水
可以显示Epoch进度

Colab中使用:
!pip install -q torch scikit-learn pandas numpy requests tqdm
!curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models_gpu.py | python
"""

import os
import warnings
warnings.filterwarnings('ignore')

print("[*] 安装中...")
os.system("pip install -q torch scikit-learn pandas numpy requests tqdm")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle

print("\n" + "="*60)
print("BB通道AI预测模型 GPU训练脚本")
print("="*60)

# ==================== 配置 ====================
HF_DATASET = "zongowo111/v2-crypto-ohlcv-data"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["15m", "1h"]
LOOKBACK = 50
FORECAST_HORIZON = 1

# GPU配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[*] 使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    昺存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"    [⚠] 没有找到GPU，使用CPU")

# 训练參数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

class BBLSTMModel(nn.Module):
    """
    LSTM模型预测BB通道
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(BBLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

def load_from_hf(symbol, timeframe):
    """从HuggingFace加载数据"""
    filename = f"klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{filename}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        return df
    except Exception as e:
        print(f"  [\u2717] 下载失败: {str(e)}")
        return None

def calculate_bb(prices, period=20, std=2):
    """计算Bollinger Band"""
    sma = prices.rolling(window=period).mean()
    std_val = prices.rolling(window=period).std()
    upper = sma + std_val * std
    lower = sma - std_val * std
    return upper, lower, sma

def prepare_sequences(df, lookback=50):
    """准备LSTM序列数据"""
    # 计算BB
    df['bb_upper'], df['bb_lower'], df['bb_middle'] = calculate_bb(df['close'])
    
    # 特征和目标
    features = []
    targets_upper = []
    targets_lower = []
    targets_middle = []
    
    # 归一化
    scaler = MinMaxScaler()
    price_data = df[['open', 'high', 'low', 'close', 'volume']].values
    price_scaled = scaler.fit_transform(price_data)
    
    for i in range(lookback, len(df) - 1):
        feature_window = price_scaled[i-lookback:i]
        features.append(feature_window)
        targets_upper.append(df.iloc[i+1]['bb_upper'])
        targets_lower.append(df.iloc[i+1]['bb_lower'])
        targets_middle.append(df.iloc[i+1]['bb_middle'])
    
    return np.array(features), np.array(targets_upper), np.array(targets_lower), np.array(targets_middle), scaler

def train_lstm_model(X, y, symbol, timeframe, metric_name, epochs=20):
    """
    训练LSTM模型
    """
    # 数据转换为PyTorch张量
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 创建模型
    model = BBLSTMModel(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1)
    model = model.to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    print(f"\n  开始训练 {symbol} {metric_name}...")
    print(f"  设备: {DEVICE}")
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")
    print(f"  批大小: {BATCH_SIZE}, Epochs: {epochs}\n")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Epoch循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 進度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}", leave=False)
        
        batch_count = 0
        for batch_X, batch_y in pbar:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # 更新進度条
            avg_loss = train_loss / batch_count
            pbar.set_postfix({'train_loss': f'{avg_loss:.6f}'})
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 打印结果
        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'{symbol}_{timeframe}_{metric_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [*] 早停: 验证损失在{patience}个epoch内没有改進")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'{symbol}_{timeframe}_{metric_name}_best.pth'))
    
    print(f"  [\u2713] 训练完成，最佳验证损失: {best_val_loss:.6f}")
    
    return model

def main():
    print("\n开始训练...\n")
    
    for symbol in SYMBOLS:
        print(f"\n=== {symbol} ===")
        
        for timeframe in TIMEFRAMES:
            print(f"  {timeframe}:")
            
            # 加载数据
            print(f"  [\u2713] 从HF下载: klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet")
            df = load_from_hf(symbol, timeframe)
            
            if df is None or len(df) < 100:
                print(f"  [\u2717] 跳过 {symbol} {timeframe}\n")
                continue
            
            print(f"  [\u2713] 加载成功: {len(df)} 根K棒")
            
            # 准备数据
            print(f"  [\u2713] 准备序列数据...")
            X, y_upper, y_lower, y_middle, scaler = prepare_sequences(df, LOOKBACK)
            print(f"  [\u2713] 特征准备: {len(X)} 整合")
            
            # 训练模型
            model_upper = train_lstm_model(X, y_upper, symbol, timeframe, 'upper', EPOCHS)
            model_lower = train_lstm_model(X, y_lower, symbol, timeframe, 'lower', EPOCHS)
            model_middle = train_lstm_model(X, y_middle, symbol, timeframe, 'middle', EPOCHS)
            
            # 保存模型
            print(f"  [\u2713] 保存模型...")
            torch.save(model_upper.state_dict(), f'{symbol}_{timeframe}_upper.pth')
            torch.save(model_lower.state_dict(), f'{symbol}_{timeframe}_lower.pth')
            torch.save(model_middle.state_dict(), f'{symbol}_{timeframe}_middle.pth')
            pickle.dump(scaler, open(f'{symbol}_{timeframe}_scaler.pkl', 'wb'))
            
            print(f"  [\u2713] 保存完成\n")
    
    print("\n" + "="*60)
    print("[\u2713] 所有模型训练完成！")
    print("="*60)

if __name__ == "__main__":
    main()