# Colab 訓練指南

## 快速開始

### 方法1: 使用curl遠端執行（推薦）

在Colab Notebook中運行：

```bash
# 安裝依賴
!pip install -q pandas scikit-learn pyarrow requests huggingface-hub

# 遠端執行訓練腳本
!curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models.py | python
```

### 方法2: 逐步手動執行

#### 步驟1: 安裝依賴

```python
!pip install pandas scikit-learn pyarrow requests huggingface-hub
```

#### 步驟2: 設置環境變數

```python
import os
os.environ['HF_TOKEN'] = 'your_hf_token_here'
os.environ['SYMBOLS'] = 'BTCUSDT,ETHUSDT'
os.environ['TIMEFRAMES'] = '15m,1h'
```

#### 步驟3: 下載訓練代碼

```python
import urllib.request

# 下載訓練腳本
url = 'https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models.py'
urllib.request.urlretrieve(url, 'train_bb_models.py')

# 執行
exec(open('train_bb_models.py').read())
```

#### 步驟4: 生成預測

```python
# 下載預測腳本
url = 'https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/generate_predictions.py'
urllib.request.urlretrieve(url, 'generate_predictions.py')

# 執行
exec(open('generate_predictions.py').read())
```

## 完整Colab Notebook範例

```python
# ==================== 初始化 ====================
!pip install -q pandas scikit-learn pyarrow requests huggingface-hub

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests
from io import BytesIO
import pyarrow.parquet as pq

# ==================== 配置 ====================
HF_TOKEN = 'your_hf_token_here'  # 替換為你的token
HF_DATASET = 'zongowo111/v2-crypto-ohlcv-data'
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15m', '1h']

# ==================== 從HF加載數據 ====================
def load_from_hf(symbol, timeframe):
    filename = f"klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{filename}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        print(f"✓ {symbol} {timeframe}: {len(df)} 根K棒")
        return df
    except Exception as e:
        print(f"✗ {symbol} {timeframe} 加載失敗: {e}")
        return None

# ==================== 訓練模型 ====================
for symbol in SYMBOLS:
    for timeframe in TIMEFRAMES:
        print(f"\n訓練 {symbol} {timeframe}...")
        
        df = load_from_hf(symbol, timeframe)
        if df is None:
            continue
        
        # 計算BB
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # 準備特徵和目標
        lookback = 50
        X, y_upper, y_lower, y_middle = [], [], [], []
        
        for i in range(lookback, len(df) - 1):
            feature = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values.flatten()
            X.append(feature)
            y_upper.append(df.iloc[i+1]['bb_upper'])
            y_lower.append(df.iloc[i+1]['bb_lower'])
            y_middle.append(df.iloc[i+1]['bb_middle'])
        
        X = np.array(X)
        y_upper = np.array(y_upper)
        y_lower = np.array(y_lower)
        y_middle = np.array(y_middle)
        
        # 訓練模型
        print(f"  特徵數: {len(X)}")
        
        model_upper = RandomForestRegressor(n_estimators=100, max_depth=10)
        model_lower = RandomForestRegressor(n_estimators=100, max_depth=10)
        model_middle = RandomForestRegressor(n_estimators=100, max_depth=10)
        
        model_upper.fit(X, y_upper)
        model_lower.fit(X, y_lower)
        model_middle.fit(X, y_middle)
        
        print(f"✓ 訓練完成")
        
        # 预测
        last_features = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].values.flatten().reshape(1, -1)
        pred_upper = model_upper.predict(last_features)[0]
        pred_lower = model_lower.predict(last_features)[0]
        pred_middle = model_middle.predict(last_features)[0]
        
        print(f"  上軌: {pred_upper:.2f}")
        print(f"  下軌: {pred_lower:.2f}")
        print(f"  中軌: {pred_middle:.2f}")
        
        # 保存模型
        import pickle
        with open(f'{symbol}_{timeframe}_upper.pkl', 'wb') as f:
            pickle.dump(model_upper, f)
        with open(f'{symbol}_{timeframe}_lower.pkl', 'wb') as f:
            pickle.dump(model_lower, f)
        with open(f'{symbol}_{timeframe}_middle.pkl', 'wb') as f:
            pickle.dump(model_middle, f)
        
        print(f"✓ 模型已保存")

print("\n✓ 所有模型訓練完成！")
print("\n接下來:")
print("1. 從Colab下載所有.pkl文件")
print("2. 上傳到 HuggingFace (zongowo111/bb-channel-models)")
print("3. 設置GitHub Action定時生成預測")
```

## 模型上傳到HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()

# 創建倉庫（如果還沒有）
repo_url = api.create_repo(
    repo_id="bb-channel-models",
    token=HF_TOKEN,
    exist_ok=True
)

print(f"倉庫URL: {repo_url}")

# 上傳所有模型
import glob
for model_file in glob.glob('*.pkl'):
    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo=f"models/{model_file}",
        repo_id="bb-channel-models",
        token=HF_TOKEN,
    )
    print(f"✓ 上傳 {model_file}")
```

## 故障排除

### 1. 無法連接HuggingFace

```python
from huggingface_hub import login
login(token='your_hf_token')
```

### 2. 數據加載超時

使用代理或增加超時時間：

```python
requests.get(url, timeout=60, proxies={'https': 'your_proxy'})
```

### 3. 記憶體不足

減少`lookback`參數或使用更小的數據集。

## 預期輸出

```
======================================================
BB通道AI預測模型 訓練腳本
======================================================

=== BTCUSDT ===
  15m:
    [✓] 加載數據：10000 根K棒
    [✓] 特徵準備：9950 整合
    [✓] 訓練模型...
    [✓] 訓練完成
      預測上軌: 45123.50
      預測下軌: 42456.25
      預測中軌: 43789.88
    [✓] 模型存儲中...
    
  1h:
    ...

======================================================
[✓] 訓練完成！
======================================================
```