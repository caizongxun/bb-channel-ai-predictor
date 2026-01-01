# GitHub Secret 设置详细指南

## ❌ 错误的地方
不是在个人设置 (https://github.com/settings/...)

## ✅ 正确的地方
是在 **Repo仓库的设置** 里

---

## 完整步骤

### 步骤1: 打开你的GitHub Repo
```
访问: https://github.com/YOUR_USERNAME/bb-channel-ai-predictor
```

### 步骤2: 进入Repo的Settings
```
在GitHub Repo页面上，找到顶部导航栏:

[Code] [Issues] [Pull requests] [Discussions] [Actions] ...

最右边找 [Settings] (齿轮图标)
点击它
```

**重要**: 这是 Repo Settings，不是个人Settings！

---

## 步骤3: 找到 Secrets and variables

进入Settings后，左侧菜单会显示:

```
左侧菜单:
├─ General
├─ Access
│  ├─ Collaborators
│  └─ ...
├─ Code and automation
│  ├─ Actions
│  ├─ Webhooks
│  └─ ...
├─ Security
│  ├─ Secrets and variables  ← 就是这个！
│  └─ ...
└─ ...
```

**找到:** "Secrets and variables" → 点击它

---

## 步骤4: 选择 Actions

进入"Secrets and variables"后，会看到三个标签:

```
[Secrets]  [Dependabot]  [Environment secrets]
         ↑
    选择这个
```

**点击:** "Secrets" 标签

---

## 步骤5: 添加新Secret

点击 "Secrets" 后，右上角会看到:

```
[New repository secret] ← 点击这个绿色按钮
```

---

## 步骤6: 填写Secret信息

### 6.1 Name字段
```
Name: HF_TOKEN
```
**必须是大写HF_TOKEN**

### 6.2 Value字段
```
1. 先获取你的HF Token
   - 访问 https://huggingface.co/settings/tokens
   - 点击 "New token"
   - 选择 "Read" 权限 (不需要Write)
   - 复制生成的token (看起来像: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)

2. 粘贴到Value字段
```

### 6.3 保存
```
点击 "Add secret" 按钮
```

---

## 验证成功

保存后，应该看到:

```
在 "Secrets" 列表中:

✓ HF_TOKEN (updated 1 second ago)
  ^^^^^^^^^                ^
  Secret名称              时间戳
```

如果能看到这个，说明成功了！

---

## 完整的左侧菜单位置

### 方法1: 直接URL
```
https://github.com/YOUR_USERNAME/bb-channel-ai-predictor/settings/secrets/actions
```

### 方法2: 从Repo页面找
```
1. 打开Repo主页
2. 点击上面的 [Settings] 标签
3. 左侧菜单找 "Security" 部分
4. 点击 "Secrets and variables"
5. 点击 "Secrets" 标签
6. 点击 "New repository secret"
```

---

## 常见错误

### ❌ 错误1: 找不到Settings标签
```
问题: 在Code/Issues/PR标签栏找不到Settings

原因: 可能你没有admin权限
      或者这是别人的Repo

解决: 确保在 YOUR_USERNAME 的Repo中
      如果是fork的，要在自己fork的版本中设置
```

### ❌ 错误2: 看不到 "Secrets and variables"
```
问题: Settings中没有这个选项

原因: 可能在其他菜单中
      或者需要向下滚动

解决: 在左侧菜单找 "Security" 部分
      Secrets在这个部分下面
```

### ❌ 错误3: Secret名称拼错
```
❌ 错误: hf_token, hf-token, HFTOKEN
✅ 正确: HF_TOKEN

必须完全匹配！
```

---

## 重要事项

### ✓ Secret会被隐藏
```
一旦保存，你无法再看到token的值
只能看到名称和更新时间
```

### ✓ 只有Actions可以读取
```
GitHub Actions会自动读取
你的个人代码看不到
这很安全！
```

### ✓ 可以更新
```
如果token过期:
1. 点击Secret名称
2. 点击 "Update"
3. 粘贴新token
4. 保存
```

---

## 完整视觉指南

```
你的Repo页面
   ↓
[Settings] 标签 (最右边)
   ↓
左侧菜单 → "Security" → "Secrets and variables"
   ↓
点击 "Secrets" 标签
   ↓
右上角 [New repository secret] 按钮
   ↓
Name: HF_TOKEN
Value: [粘贴你的HF token]
   ↓
点击 "Add secret"
   ↓
✓ 看到 HF_TOKEN (updated 1 second ago)
   ↓
完成！
```

---

## 快速检查清单

```
□ 在Repo的Settings中（不是个人Settings）
□ 进入 Secrets and variables
□ 点击 "Secrets" 标签
□ Name 填写: HF_TOKEN
□ Value 填写: 你的HF token
□ 点击 "Add secret"
□ 看到 HF_TOKEN 在列表中
```

---

## 直接URL方法（最快）

替换 `YOUR_USERNAME` 为你的GitHub用户名:

```
https://github.com/YOUR_USERNAME/bb-channel-ai-predictor/settings/secrets/actions
```

直接访问这个URL就能到设置页面！