# GPU 训练 - 快速开始指南

## 🚀 三步快速开始

### 步骤1: 在Colab选择GPU
```
菜单 → 运行时 → 更改运行时类型
    ↓
选择: GPU (T4 或 A100)
    ↓
点击 "保存"
```

### 步骤2: 运行GPU训练脚本
```bash
# 复制粘贴到Colab中运行
!pip install -q torch scikit-learn pandas numpy requests tqdm
!curl -s https://raw.githubusercontent.com/caizongxun/bb-channel-ai-predictor/main/colab_scripts/train_bb_models_gpu.py | python
```

### 步骤3: 等待训练完成
```
你会看到:

[*] 使用设备: cuda:0
    GPU: Tesla T4
    昺存: 16.00 GB

=== BTCUSDT ===
  15m:
  [✓] 从HF下载: klines/BTCUSDT/BTC_15m.parquet
  [✓] 加载成功: 219643 根K棒
  [✓] 准备序列数据...
  [✓] 特征准备: 219592 整合
  
  开始训练 BTCUSDT upper...
  设备: cuda:0
  训练集: 175936, 验证集: 43984
  批大小: 32, Epochs: 20
  
  Epoch  1/20: ████████████▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌ 40% [train_loss: 0.001234]
  Epoch  1/20: ████████████████████████████████ 100% [train_loss: 0.000567]
  Epoch  1/20 | Train Loss: 0.000567 | Val Loss: 0.000489
  Epoch  2/20 | Train Loss: 0.000456 | Val Loss: 0.000398
  Epoch  3/20 | Train Loss: 0.000345 | Val Loss: 0.000287
  ...
```

---

## 📊 输出说明

### Epoch进度条
```
Epoch  1/20: ████████████▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌ 40% [train_loss: 0.001234]
       ↑                ↑                      ↑      ↑
    Epoch号       进度条 (40%)            完成度   当前批损失
```

### 每个 Epoch 的结果
```
Epoch  1/20 | Train Loss: 0.000567 | Val Loss: 0.000489
       ↑                ↑                  ↑
    Epoch号      训练集平均损失         验证集损失
```

**更小的Loss = 更好的模型**

---

## ⚡ 性能对比

### 训练速度 (219K数据)

| 方式 | 时间 | Loss |
|------|------|------|
| **RandomForest (CPU)** | ~5分钟 | 较高 |
| **LSTM (CPU)** | ~15分钟 | 较低 |
| **LSTM (GPU)** | **~1-2分钟** | **最低** |

**GPU快7-10倍！**

---

## 📁 生成的文件

训练完成后，会生成:

```
BTCUSDT_15m_upper.pth       ← 上轨模型
BTCUSDT_15m_lower.pth       ← 下轨模型
BTCUSDT_15m_middle.pth      ← 中轨模型
BTCUSDT_15m_scaler.pkl      ← 数据归一化器

ETHUSDT_15m_upper.pth
ETHUSDT_15m_lower.pth
ETHUSDT_15m_middle.pth
ETHUSDT_15m_scaler.pkl

...
```

---

## 📥 下载模型

### 方法1: 直接下载
```python
# 在Colab中运行
from google.colab import files

# 选择要下载的文件
files.download('BTCUSDT_15m_upper.pth')
files.download('BTCUSDT_15m_lower.pth')
files.download('BTCUSDT_15m_middle.pth')
files.download('BTCUSDT_15m_scaler.pkl')
# 其他币种...
```

---

## 🔧 自定义參数

### 快速训练 (1分钟)
```python
BATCH_SIZE = 64
EPOCHS = 5
HIDDEN_SIZE = 32
```

### 高精度训练 (3分钟)
```python
BATCH_SIZE = 16
EPOCHS = 50
HIDDEN_SIZE = 128
NUM_LAYERS = 3
```

---

## ⚠️ 常见问题

### Q1: "CUDA out of memory"
**原因**: GPU昺存不足

**解决**:
```python
# 减小BATCH_SIZE
BATCH_SIZE = 16  # 改小

# 或减小隐层大小
HIDDEN_SIZE = 32  # 改小
```

### Q2: "No CUDA devices available"
**原因**: Colab没有选GPU

**解决**: 
重新选择运行时 → GPU → 保存 → 重启

### Q3: Loss没有下降
**原因**: 学习率太高

**解决**:
```python
# 降低学习率
LEARNING_RATE = 0.0001  # 改小

# 或增加训练轮数
EPOCHS = 50  # 改大
```

---

## 🏆 成功标志

当你看到:
```
============================================================
[✓] 所有模型训练完成！
============================================================
```

说明训练成功了！

接下来继续按照原来的流程:
1. 上传到HF
2. 配置GitHub Secret
3. 手动运行GitHub Action
4. 在TradingView加载

---

**现在就在Colab中运行GPU训练吧！🚀**