"""
預測生成腳本 - 想序一步步的預測未來BB通道
可以被 GitHub Action 或 Colab 訃用
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# ==================== 配置 ====================
HF_TOKEN = os.getenv('HF_TOKEN', '')
HF_DATASET = 'zongowo111/v2-crypto-ohlcv-data'
HF_MODEL_REPO = 'zongowo111/bb-channel-models'

SYMBOLS = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
TIMEFRAME = os.getenv('TIMEFRAME', '15m')  # GitHub Action 會傳入

OUTPUT_DIR = 'predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"BB通道預測生成 - {TIMEFRAME}時間框架")
print(f"{'='*60}")
print(f"\u5e63種: {', '.join(SYMBOLS)}")
print(f"\u6642間框架: {TIMEFRAME}")
print()

def load_from_hf(symbol, timeframe):
    """從HuggingFace加載Parquet数據"""
    filename = f"klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{filename}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        
        # 確保時間序列
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        return df
    except Exception as e:
        print(f"  [×] 下載失敗: {str(e)}")
        return None

def calculate_bb(df, period=20, std=2):
    """計算Bollinger Band"""
    sma = df['close'].rolling(window=period).mean()
    std_val = df['close'].rolling(window=period).std()
    upper = sma + std_val * std
    lower = sma - std_val * std
    return upper, lower, sma

def load_model(symbol, metric):
    """從HuggingFace或本地加載模型"""
    # 先简單地，從本地.pkl加載
    # 實現時可以修改为HF下載
    model_path = f"{symbol}_{TIMEFRAME}_{metric}.pkl"
    
    if not os.path.exists(model_path):
        print(f"  [✗] 找不到模型: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"  [×] 加載模型失敢: {str(e)}")
        return None

def generate_predictions(symbol, timeframe):
    """預測未來的BB通道"""
    print(f"\n{symbol} ({timeframe}):")
    
    # 1. 加載數據
    print(f"  [⋉] 加載數據...")
    df = load_from_hf(symbol, timeframe)
    if df is None:
        return None
    
    print(f"  [✓] 加載 {len(df)} 根K棒")
    
    # 2. 計算當前BB
    print(f"  [⋉] 計算當前BB...")
    bb_upper, bb_lower, bb_middle = calculate_bb(df)
    
    # 3. 加載模型
    print(f"  [⋉] 加載模型...")
    model_upper = load_model(symbol, 'upper')
    model_lower = load_model(symbol, 'lower')
    model_middle = load_model(symbol, 'middle')
    
    if not all([model_upper, model_lower, model_middle]):
        print(f"  [×] 找不到模型，跳過此幣種")
        return None
    
    # 4. 預測
    print(f"  [⋉] 預測...")
    lookback = 50
    
    if len(df) < lookback:
        print(f"  [×] 数據不足 ({len(df)} < {lookback})")
        return None
    
    # 最後50根K棒的特徵
    last_features = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].values.flatten().reshape(1, -1)
    
    pred_upper = model_upper.predict(last_features)[0]
    pred_lower = model_lower.predict(last_features)[0]
    pred_middle = model_middle.predict(last_features)[0]
    
    # 生成預測結果
    predictions = {
        'timestamp': datetime.utcnow().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'current_price': float(df['close'].iloc[-1]),
        'bb_upper': float(pred_upper),
        'bb_lower': float(pred_lower),
        'bb_middle': float(pred_middle),
        'actual_bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
        'actual_bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
        'actual_bb_middle': float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
    }
    
    print(f"  [✓] 預測一接隋成")
    print(f"      預測上軌: {pred_upper:.2f}")
    print(f"      預測下軌: {pred_lower:.2f}")
    print(f"      預測中軌: {pred_middle:.2f}")
    
    return predictions

def save_predictions(all_predictions):
    """保存預測为CSV和JSON"""
    if not all_predictions:
        return
    
    # 保存為一个大CSV文件
    df = pd.DataFrame(all_predictions)
    csv_file = os.path.join(OUTPUT_DIR, f"predictions_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n[✓] 預測已保存至: {csv_file}")
    
    # 保存為旅一个JSON文件
    json_file = os.path.join(OUTPUT_DIR, f"predictions_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(json_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"[✓] JSON已保存至: {json_file}")

def main():
    all_predictions = []
    
    for symbol in SYMBOLS:
        pred = generate_predictions(symbol, TIMEFRAME)
        if pred:
            all_predictions.append(pred)
    
    if all_predictions:
        save_predictions(all_predictions)
        print(f"\n{"="*60}")
        print(f"[✓] 預測完成！")
        print(f"{"="*60}")
        return True
    else:
        print(f"\n[×] 沒有預測結果")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)