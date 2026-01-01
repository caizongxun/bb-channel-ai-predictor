# Colab Config for BB Channel Predictor

# HuggingFace
HF_DATASET_NAME = "zongowo111/v2-crypto-ohlcv-data"
HF_MODEL_REPO = "zongowo111/bb-channel-models"  # 你的HF仓庫（需托管元数據）
HF_TOKEN = "your_hf_token_here"  # 設置你的HF token

# 幣種配置
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    # 易次新增更多幣種
]

# 時間框架
 TIME_FRAMES = ["15m", "1h"]

# 模型配置
MODEL_CONFIG = {
    "model_type": "RandomForest",  # RandomForest / LSTM / GRU
    "n_estimators": 100,
    "max_depth": 10,
    "lookback": 50,  # 使用運最恦50根患線作為輸入
    "forecast_horizon": 1,  # 預測未來 1 根K棵
}

# 訓練參數
TRAIN_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "epochs": 10,  # 統計学習模型的epoch
    "batch_size": 32,
}

# HuggingFace 的Parquet檔路徑格式
PARQUET_PATH_TEMPLATE = "klines/{symbol}/{symbol.replace('USDT', '')}_{timeframe}.parquet"

# 輸出檔案名稱
OUTPUT_FILE_TEMPLATE = "{symbol}_{timeframe}_predictions.csv"
MODEL_FILE_TEMPLATE = "{symbol}_{timeframe}_bb_model.pkl"

# 預測配置
PREDICTION_CONFIG = {
    "future_days": 5,  # 預測未來 5 天
    "update_frequency": "15m",  # 15m 或 1h
}

# Log配置
LOG_LEVEL = "INFO"
VERBOSE = True