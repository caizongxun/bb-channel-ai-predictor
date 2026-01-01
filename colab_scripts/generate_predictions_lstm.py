"""
LSTM预测脚本 - 根据训练好的GPU LSTM模型预测

GitHub Action 或 Colab 中使用:
curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/generate_predictions_lstm.py | python
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("[*] 安装中...")
os.system("pip install -q torch scikit-learn pandas numpy requests tqdm")

import torch
import torch.nn as nn

# ==================== 配置 ====================
HF_MODEL_REPO = 'zongowo111/bb-channel-models'
HF_DATASET = 'zongowo111/v2-crypto-ohlcv-data'

SYMBOLS = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
TIMEFRAME = os.getenv('TIMEFRAME', '15m')

OUTPUT_DIR = 'predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n[*] 使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

print(f"\n" + "="*60)
print(f"LSTM BB通道预测生成 - {TIMEFRAME}時間框架")
print("="*60)
print(f"\n币种: {', '.join(SYMBOLS)}")
print(f"时间框架: {TIMEFRAME}")
print()

# ==================== LSTM模型定义 ====================
class BBLSTMModel(nn.Module):
    """
    LSTM模型预测BB通道
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(BBLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

# ==================== 方法 ====================
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
        print(f"  [✗] 下载失败: {str(e)}")
        return None

def load_model_from_hf(symbol, timeframe, metric):
    """从HuggingFace下载模型"""
    model_file = f"{symbol}_{timeframe}_{metric}_best.pth"
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/models/{model_file}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存为临时文件
        with open(model_file, 'wb') as f:
            f.write(response.content)
        
        # 加载模型
        model = BBLSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # 删除临时文件
        os.remove(model_file)
        
        return model
    except Exception as e:
        print(f"  [✗] 加载model失败 {metric}: {str(e)}")
        return None

def load_scaler_from_hf(symbol, timeframe):
    """从HuggingFace下载scaler"""
    scaler_file = f"{symbol}_{timeframe}_scaler.pkl"
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/scalers/{scaler_file}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存为临时文件
        with open(scaler_file, 'wb') as f:
            f.write(response.content)
        
        # 加载scaler
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        # 删除临时文件
        os.remove(scaler_file)
        
        return scaler
    except Exception as e:
        print(f"  [✗] 加载scaler失败: {str(e)}")
        return None

def calculate_bb(prices, period=20, std=2):
    """计算Bollinger Band"""
    sma = prices.rolling(window=period).mean()
    std_val = prices.rolling(window=period).std()
    upper = sma + std_val * std
    lower = sma - std_val * std
    return upper, lower, sma

def predict_with_lstm(symbol, timeframe):
    """使用LSTM模型預测未來的BB通道"""
    print(f"\n{symbol} ({timeframe}):")
    
    # 1. 加载数据
    print(f"  [•] 加载数据...")
    df = load_from_hf(symbol, timeframe)
    if df is None:
        return None
    
    print(f"  [✓] 加载 {len(df)} 根K棒")
    
    # 2. 计算当前BB
    print(f"  [•] 计算当前BB...")
    bb_upper, bb_lower, bb_middle = calculate_bb(df['close'])
    
    # 3. 下载scaler
    print(f"  [•] 加载scaler...")
    scaler = load_scaler_from_hf(symbol, timeframe)
    if scaler is None:
        return None
    
    # 4. 准备最后的特征
    print(f"  [•] 准备特征...")
    lookback = 50
    
    if len(df) < lookback:
        print(f"  [✗] 数据不足")
        return None
    
    # 归一化最后的lookback篇k棒
    price_data = df[['open', 'high', 'low', 'close', 'volume']].values
    price_scaled = scaler.fit_transform(price_data)
    
    last_features = price_scaled[-lookback:].reshape(1, lookback, 5)
    
    # 5. 加载模型并预测
    print(f"  [•] 加载模型并预测...")
    
    X_tensor = torch.FloatTensor(last_features).to(DEVICE)
    
    # 上轈, 下轨, 中轨的預测
    predictions = {}
    
    for metric in ['upper', 'lower', 'middle']:
        model = load_model_from_hf(symbol, timeframe, metric)
        if model is None:
            print(f"  [✗] 无法加载{metric}模型")
            return None
        
        with torch.no_grad():
            output = model(X_tensor).cpu().numpy()[0][0]
        
        predictions[metric] = float(output)
    
    print(f"  [✓] 预测完成")
    print(f"      上轨: {predictions['upper']:.2f}")
    print(f"      下轨: {predictions['lower']:.2f}")
    print(f"      中轨: {predictions['middle']:.2f}")
    
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'current_price': float(df['close'].iloc[-1]),
        'bb_upper': predictions['upper'],
        'bb_lower': predictions['lower'],
        'bb_middle': predictions['middle'],
        'actual_bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
        'actual_bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
        'actual_bb_middle': float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
    }

def main():
    all_predictions = []
    
    for symbol in SYMBOLS:
        pred = predict_with_lstm(symbol, TIMEFRAME)
        if pred:
            all_predictions.append(pred)
    
    if all_predictions:
        # 保存为CSV
        df = pd.DataFrame(all_predictions)
        csv_file = os.path.join(OUTPUT_DIR, f"predictions_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n[✓] 预测已保存至: {csv_file}")
        
        # 保存为JSON
        json_file = os.path.join(OUTPUT_DIR, f"predictions_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(json_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"[✓] JSON已保存至: {json_file}")
        
        print(f"\n" + "="*60)
        print(f"[✓] 预测完成！")
        print("="*60)
        return True
    else:
        print(f"\n[×] 没有预测结果")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)