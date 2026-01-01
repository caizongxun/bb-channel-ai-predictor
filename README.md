# BB Channel AI Predictor

AI-powered Bollinger Band channel prediction system for cryptocurrency trading with TradingView integration.

## 系統架構

```
你的HuggingFace數據
    ↓
Colab遠端訓練
    ↓
模型保存到HF
    ↓
GitHub Action定時更新預測
    ↓
TradingView讀取預測
    ↓
自動交易信號
```

## 項目結構

```
bb-channel-ai-predictor/
├── colab_scripts/
│   ├── train_bb_models.py          # 模型訓練主腳本
│   ├── generate_predictions.py     # 預測生成腳本
│   └── config.py                   # 配置文件
├── pine_script/
│   ├── bb_predictor_with_ai.pine   # TradingView指標
│   └── entry_signal_ai_integrated.pine  # 完整進場信號
├── .github/workflows/
│   ├── predict_15m.yml             # 15分鐘時間框架預測
│   ├── predict_1h.yml              # 1小時時間框架預測
│   └── on_demand.yml               # 按需預測（幣種變更時）
├── requirements.txt                # Python依賴
└── README.md                        # 本文件
```

## 功能特性

- ✓ 使用HuggingFace遠端數據訓練
- ✓ 每個幣種每個時間框架獨立模型
- ✓ 自動化預測和上傳
- ✓ TradingView seamless集成
- ✓ 支持動態幣種切換
- ✓ 綠色標記準確預測區域

## 快速開始

### 1. 訓練模型（Colab）

打開Colab notebook並運行：
```bash
# 遠端執行訓練（無需克隆倉庫）
curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models.py | python
```

### 2. 生成預測

```bash
curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/generate_predictions.py | python
```

### 3. TradingView集成

複製 `pine_script/entry_signal_ai_integrated.pine` 到TradingView編輯器

## 配置說明

見 `colab_scripts/config.py`

## 數據源

- HuggingFace: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- Binance 15m, 1h 數據
- 支持所有主流幣種

## 更新頻率

- **15分鐘框架**: 每15分鐘自動更新一次
- **1小時框架**: 每小時自動更新一次
- **按需**: 幣種或時間框架變更時立即更新

## 貢獻

歡迎提交Issue和PR

## 許可

MIT