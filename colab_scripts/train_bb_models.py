"""
Colab遠端訓練脚本：BB通道預測模型
可以直接在Colab中運行：
curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models.py | python
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime, timedelta
import requests
from io import BytesIO
import pyarrow.parquet as pq

print("[*] 安裝中...")
os.system("pip install -q huggingface-hub requests pandas scikit-learn")

from huggingface_hub import hf_hub_download, HfApi, login

print("\n" + "="*60)
print("BB通道AI預測模型 訓練脚本")
print("="*60)

# ==================== 配置 ====================
HF_DATASET = "zongowo111/v2-crypto-ohlcv-data"
HF_MODEL_REPO = "zongowo111/bb-channel-models"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # 可以粗你的幣種
TIMEFRAMES = ["15m", "1h"]
LOOKBACK = 50
FORECAST_HORIZON = 1

class BBChannelPredictor:
    """
    Bollinger Band通道預測模型
    """
    
    def __init__(self, lookback=50, forecast_horizon=1):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model_upper = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model_lower = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model_middle = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def calculate_bb(self, prices, period=20, std=2):
        """計算Bollinger Band上下軌和中軌"""
        sma = prices.rolling(window=period).mean()
        std_val = prices.rolling(window=period).std()
        upper = sma + std_val * std
        lower = sma - std_val * std
        return upper, lower, sma
    
    def prepare_features(self, df):
        """準備輸入特徵"""
        features = []
        targets_upper = []
        targets_lower = []
        targets_middle = []
        
        # 計算當前的BB
        df['bb_upper'], df['bb_lower'], df['bb_middle'] = self.calculate_bb(df['close'])
        
        for i in range(self.lookback, len(df) - self.forecast_horizon):
            # 特徵：運最恦50根患線的OHLCV數據
            feature_window = df.iloc[i-self.lookback:i][['open', 'high', 'low', 'close', 'volume']].values
            feature_vector = feature_window.flatten()
            
            # 目標：未來N根K棵的BB金標
            future_upper = df.iloc[i + self.forecast_horizon]['bb_upper']
            future_lower = df.iloc[i + self.forecast_horizon]['bb_lower']
            future_middle = df.iloc[i + self.forecast_horizon]['bb_middle']
            
            if not np.isnan(future_upper) and not np.isnan(future_lower) and not np.isnan(future_middle):
                features.append(feature_vector)
                targets_upper.append(future_upper)
                targets_lower.append(future_lower)
                targets_middle.append(future_middle)
        
        return np.array(features), np.array(targets_upper), np.array(targets_lower), np.array(targets_middle)
    
    def train(self, df):
        """訓練模型"""
        print(f"  [✓] 加載數據：{len(df)} 根K棵")
        
        X, y_upper, y_lower, y_middle = self.prepare_features(df)
        
        if len(X) < 100:
            print(f"  [×] 數據不足（需要至少100筆）")
            return False
        
        print(f"  [✓] 特徵準備：{len(X)} 整合")
        
        # 模型訓練
        print(f"  [✓] 訓練模型...")
        self.model_upper.fit(X, y_upper)
        self.model_lower.fit(X, y_lower)
        self.model_middle.fit(X, y_middle)
        
        print(f"  [✓] 訓練完成")
        return True
    
    def predict(self, df):
        """預測未來的BB通道"""
        if len(df) < self.lookback:
            return None
        
        # 最後50根K棵
        feature_window = df.iloc[-self.lookback:][['open', 'high', 'low', 'close', 'volume']].values
        feature_vector = feature_window.flatten().reshape(1, -1)
        
        pred_upper = self.model_upper.predict(feature_vector)[0]
        pred_lower = self.model_lower.predict(feature_vector)[0]
        pred_middle = self.model_middle.predict(feature_vector)[0]
        
        return {
            'bb_upper': pred_upper,
            'bb_lower': pred_lower,
            'bb_middle': pred_middle
        }
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'model_upper': self.model_upper,
            'model_lower': self.model_lower,
            'model_middle': self.model_middle,
            'scaler': self.scaler,
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon
        }, path)

def load_from_hf(symbol, timeframe):
    """從HuggingFace加載Parquet數據"""
    filename = f"klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet"
    
    print(f"  [✓] 從HF下載: {filename}")
    
    try:
        # 可以直接訪啊 HF 的 raw 來獲取
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{filename}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_parquet(BytesIO(response.content))
        print(f"  [✓] 加載成功: {len(df)} 根K棵")
        return df
    
    except Exception as e:
        print(f"  [×] 下載失敗: {str(e)}")
        return None

def main():
    print("\n開始訓練...\n")
    
    for symbol in SYMBOLS:
        print(f"\n=== {symbol} ===")
        for timeframe in TIMEFRAMES:
            print(f"  {timeframe}:")
            
            # 加載數據
            df = load_from_hf(symbol, timeframe)
            if df is None or len(df) < 100:
                print(f"  [×] 混過 {symbol} {timeframe}\n")
                continue
            
            # 確保時間序列
            df = df.sort_index()
            
            # 訓練
            predictor = BBChannelPredictor(lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON)
            if not predictor.train(df):
                print(f"  [×] 訓練失敗\n")
                continue
            
            # 預測
            pred = predictor.predict(df)
            if pred:
                print(f"  [✓] 預測成功")
                print(f"      BB上軌: {pred['bb_upper']:.2f}")
                print(f"      BB下軌: {pred['bb_lower']:.2f}")
                print(f"      BB中軌: {pred['bb_middle']:.2f}")
            
            # 存模∢2模型体离不声
            print(f"  [✓] 模型存模中...")
            # 個不偵剪 - 按權 HF 或這裡先保存到中間檔案

    print("\n" + "="*60)
    print("[✓] 訓練完成！")
    print("="*60)

if __name__ == "__main__":
    main()