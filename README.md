# 🧠 自监督神经网络 (Self-Supervised Learning) 交互式演示 Demo

> 基于 **SimCLR** 对比学习框架，使用 **PyTorch + Gradio** 构建的自监督学习可视化教学工具。

---

## 📋 目录

- [项目简介](#项目简介)
- [技术栈与依赖](#技术栈与依赖)
- [项目结构](#项目结构)
- [核心原理详解](#核心原理详解)
- [代码架构详解](#代码架构详解)
- [快速开始](#快速开始)
- [使用流程](#使用流程)
- [超参数说明](#超参数说明)
- [技术细节与设计决策](#技术细节与设计决策)
- [扩展与改进方向](#扩展与改进方向)
- [参考文献](#参考文献)

---

## 项目简介

本项目是一个 **自监督对比学习** 的交互式演示 Demo，旨在通过可视化的方式帮助用户理解自监督学习的核心概念和工作原理。

### 核心特性

| 特性 | 说明 |
|:---|:---|
| 🎯 **SimCLR 对比学习** | 实现了完整的 SimCLR 自监督学习流程 |
| 🖥️ **交互式可视化** | 基于 Gradio 的 Web 界面，支持实时交互 |
| 📊 **数据增强演示** | 可视化展示对比学习的数据增强策略 |
| 🏗️ **架构可视化** | 动态展示 SimCLR 模型架构图 |
| 📈 **训练过程监控** | 实时展示训练损失曲线和进度 |
| 🔬 **对比实验** | 对比自监督特征 vs 随机特征的效果差异 |
| 🗺️ **特征空间可视化** | t-SNE 降维可视化高维特征空间 |

### 整体工作流程

```
准备 CIFAR-10 数据 --> 数据增强可视化 --> 查看 SimCLR 架构 --> 自监督预训练(不使用标签) --> 线性评估 & 对比实验
```

---

## 技术栈与依赖

| 技术 | 版本要求 | 用途 |
|:---|:---|:---|
| **PyTorch** | >= 2.0.0 | 深度学习框架，用于模型定义与训练 |
| **TorchVision** | >= 0.15.0 | 数据集加载 (CIFAR-10) 和图像变换 |
| **Gradio** | >= 4.0.0 | Web 交互界面构建 |
| **NumPy** | >= 1.24.0 | 数值计算 |
| **Matplotlib** | >= 3.7.0 | 图表绘制与可视化 |
| **scikit-learn** | >= 1.2.0 | 线性分类器 (LogisticRegression)、t-SNE 降维 |
| **Pillow** | >= 9.4.0 | 图像处理 |

---

## 项目结构

```
自监督神经网络/
├── app.py                      # Gradio Web 应用入口（界面层）
├── self_supervised_core.py     # 核心模型与训练逻辑（业务层）
├── requirements.txt            # Python 依赖包列表
├── README.md                   # 项目技术文档（本文件）
├── data/                       # 数据集目录
│   └── cifar-10-batches-py/    # CIFAR-10 数据集文件
│       ├── batches.meta        # 元数据（类别名称等）
│       ├── data_batch_1~5      # 训练集（共5个批次）
│       └── test_batch          # 测试集
└── __pycache__/                # Python 字节码缓存
```

**架构分层**：项目采用清晰的两层架构设计：

- **界面层 (app.py)**：Gradio 界面 + 回调函数
- **业务层 (self_supervised_core.py)**：包含 SelfSupervisedTrainer（训练与评估）、SimCLRModel（模型定义）、NTXentLoss（损失函数）、SimCLRAugmentation（数据增强）、Visualizer（可视化工具）

---

## 核心原理详解

### 什么是自监督学习

自监督学习 (Self-Supervised Learning) 是一种 **不需要人工标签** 的深度学习方法。它通过设计巧妙的 **代理任务 (Pretext Task)**，让模型从数据本身学习有用的特征表示。

| 对比维度 | 监督学习 | 自监督学习 |
|:---:|:---:|:---:|
| **标签需求** | 需要大量人工标注 | 不需要标签 |
| **数据利用** | 仅能使用有标签数据 | 可利用海量无标签数据 |
| **标注成本** | 高（特别是医学等领域） | 低 |
| **典型方法** | 分类、检测、分割 | 对比学习、掩码预测、自蒸馏 |

### SimCLR 框架原理

SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) 是由 Google Brain 团队在 2020 年提出的一种简单而有效的对比学习框架。

**核心思想**：

> 对同一张图片施加不同的随机变换，得到两个"视图"(View)。网络需要学会让 **同一图片的不同视图特征相似**（正样本对），**不同图片的视图特征不同**（负样本对）。

**SimCLR 工作流程**：

```
                    随机增强 t
输入图片 x ─────────────────────→ 视图 x_i → 编码器 f → 特征 h_i → 投影头 g → 投影 z_i ─┐
     │                                                                                    │
     │                                                                               NT-Xent 损失
     │                                                                                    │
     └──────────────────────→ 视图 x_j → 编码器 f → 特征 h_j → 投影头 g → 投影 z_j ─┘
                    随机增强 t'             (共享权重)              (共享权重)
```

### 数据增强策略

数据增强是自监督对比学习的 **核心驱动力**。本项目采用 SimCLR 论文推荐的增强组合：

| 增强操作 | 参数 | 目的 |
|:---|:---|:---|
| **随机裁剪缩放** (RandomResizedCrop) | scale=(0.2, 1.0), size=32 | 学习尺度和位置不变性 |
| **随机水平翻转** (RandomHorizontalFlip) | p=0.5 | 学习方向不变性 |
| **颜色抖动** (ColorJitter) | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1，概率 p=0.8 | 学习颜色不变性 |
| **随机灰度化** (RandomGrayscale) | p=0.2 | 减少对颜色信息的过度依赖 |
| **归一化** (Normalize) | mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010] | CIFAR-10 数据集标准化参数 |

**代码实现** (`SimCLRAugmentation` 类)：

```python
class SimCLRAugmentation:
    def __init__(self, img_size=32):
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])

    def __call__(self, x):
        # 对同一张图片生成两个不同的增强视图
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2
```

每次调用时，由于随机性，同一张图片会生成两个不同的增强视图，作为对比学习的正样本对。

### 模型架构设计

SimCLR 模型由两个核心组件构成：

#### 1. 特征编码器 f (Encoder)

基于简化版 ResNet 架构，将 32×32 的彩色图片映射为 **128 维特征向量**。

```
输入 [B, 3, 32, 32]
    │
    ▼
Conv2d(3→32, 3×3) + BN + ReLU     # 初始卷积层
    │
    ▼
ResBlock(32→32, stride=1)          # 残差块1：保持空间尺寸
    │
    ▼
ResBlock(32→64, stride=2)          # 残差块2：空间下采样 32→16
    │
    ▼
ResBlock(64→128, stride=2)         # 残差块3：空间下采样 16→8
    │
    ▼
AdaptiveAvgPool2d(1×1)             # 全局平均池化
    │
    ▼
Linear(128→128)                    # 全连接层
    │
    ▼
输出 特征向量 h [B, 128]           # ← 下游任务使用此特征
```

**残差块 (SimpleResBlock)** 的设计：

```
输入 x ──────────────────────┐
   │                         │
   ▼                         │ (shortcut 残差连接)
Conv2d + BN + ReLU           │
   │                         │
   ▼                         │
Conv2d + BN                  │
   │                         │
   ▼                         ▼
   └─── 逐元素相加 ──────── + ── ReLU ──→ 输出
```

残差连接使得梯度更容易流动，缓解深层网络的梯度消失问题。

#### 2. 投影头 g (ProjectionHead)

一个简单的两层 MLP，将编码器特征进一步映射到对比学习空间。

```
输入 h [B, 128]
    │
    ▼
Linear(128→128) + ReLU        # 隐藏层
    │
    ▼
Linear(128→64)                 # 输出层
    │
    ▼
输出 投影 z [B, 64]            # ← 对比损失在此空间计算
```

> 🔑 **为什么需要投影头？**
> SimCLR 论文发现：在投影头输出空间计算对比损失效果更好。投影头可能会丢弃一些对对比任务无用但对下游任务有用的信息，因此下游任务使用编码器输出 h 而非投影头输出 z。**投影头在预训练完成后丢弃。**

### NT-Xent 对比损失函数

**NT-Xent** (Normalized Temperature-scaled Cross Entropy Loss)，即归一化温度缩放交叉熵损失，是 SimCLR 的核心损失函数。

#### 数学公式

给定一个 batch 中 N 张图片，每张生成 2 个视图，共 2N 个样本。对于正样本对 (i, j)：

```
L(i,j) = -log( exp(sim(z_i, z_j) / τ) / Σ_{k=1,k≠i}^{2N} exp(sim(z_i, z_k) / τ) )
```

其中：
- `sim(z_i, z_j) = (z_i · z_j) / (‖z_i‖ · ‖z_j‖)` 是余弦相似度
- `τ` 是温度参数 (temperature)
- 分子：正样本对的相似度（同一图片的两个视图）
- 分母：当前样本与所有其他样本的相似度之和

#### 计算流程

```
步骤 1: L2 归一化所有投影向量 z_i, z_j
         ↓
步骤 2: 拼接 z = [z_i; z_j]，得到 [2N, dim] 矩阵
         ↓
步骤 3: 计算余弦相似度矩阵 sim = z · z^T / τ，得到 [2N, 2N]
         ↓
步骤 4: 构造标签 —— 对于样本 i，正样本是样本 i+N（反之亦然）
         ↓
步骤 5: 屏蔽对角线（排除自身与自身的相似度）
         ↓
步骤 6: 计算交叉熵损失
```

#### 温度参数 τ 的作用

| τ 值 | 效果 | 说明 |
|:---|:---|:---|
| **较小** (如 0.1) | 分布更"尖锐" | 更关注困难负样本（与正样本相似的负样本） |
| **较大** (如 1.0) | 分布更"平滑" | 更均匀地对待所有负样本 |
| **本项目默认** | τ = 0.5 | 平衡困难样本关注度与训练稳定性 |

### 线性评估协议

线性评估 (Linear Evaluation Protocol) 是自监督学习领域评估特征质量的 **标准方法**。

**评估流程**：

```
冻结编码器 f (不更新参数) --> 提取所有样本的特征 h --> 在特征 h 上训练逻辑回归分类器 --> 评估分类准确率
```

**核心逻辑**：
1. **冻结编码器** —— 编码器参数不再更新
2. **训练线性分类器** —— 仅训练一层逻辑回归 (LogisticRegression)
3. **高准确率** → 编码器学到了语义丰富的特征
4. **低准确率** → 特征没有很好地捕获语义信息

**对比实验设置**：

| 方案 | 编码器状态 | 预期准确率 |
|:---|:---|:---|
| 随机猜测基线 | — | 10%（10个类别） |
| 随机初始化编码器 + 线性分类器 | 未训练 | 较低 |
| **自监督预训练编码器 + 线性分类器** | **SimCLR 预训练** | **显著更高** |

---

## 代码架构详解

### 核心模块 self_supervised_core.py

该文件包含了项目的全部核心逻辑，按功能划分为五个部分：

#### 各类职责说明

| 类名 | 职责 | 所属模块 |
|:---|:---|:---|
| `SimCLRAugmentation` | SimCLR 风格数据增强管道，每次调用生成两个不同的增强视图 | 数据增强 |
| `ContrastiveDataset` | 对比学习数据集包装器，将普通数据集转换为 (view1, view2, label) 格式 | 数据增强 |
| `SimpleResBlock` | 简化的残差块，包含两层卷积和跳跃连接 | 模型架构 |
| `Encoder` | 特征编码器，基于简化 ResNet，输出 128 维特征 | 模型架构 |
| `ProjectionHead` | 投影头 MLP，将特征映射到 64 维对比学习空间 | 模型架构 |
| `SimCLRModel` | 完整 SimCLR 模型 = Encoder + ProjectionHead | 模型架构 |
| `NTXentLoss` | NT-Xent 对比损失函数 | 损失函数 |
| `SelfSupervisedTrainer` | 训练器，封装数据准备、预训练、特征提取、线性评估全流程 | 训练评估 |
| `Visualizer` | 可视化工具集，负责所有图表的生成 | 可视化 |

#### 类之间的关系

```
SimCLRModel
├── Encoder (包含 3 个 SimpleResBlock)
└── ProjectionHead

SelfSupervisedTrainer
├── 使用 SimCLRModel
├── 使用 NTXentLoss
├── 使用 SimCLRAugmentation
└── 使用 ContrastiveDataset
    └── ContrastiveDataset 内部使用 SimCLRAugmentation
```

#### 关键训练流程代码解析

**自监督预训练 (`pretrain` 方法)** 的核心循环：

```python
for epoch in range(epochs):
    for view1, view2, _ in dataloader:  # 注意：标签 _ 完全没有使用！
        view1, view2 = view1.to(device), view2.to(device)

        # 前向传播：两个视图分别通过模型
        _, z1 = model(view1)  # 视图1 → 编码器 → 投影头 → 投影z1
        _, z2 = model(view2)  # 视图2 → 编码器 → 投影头 → 投影z2

        # 计算对比损失：拉近正样本对 (z1, z2)，推远负样本对
        loss = criterion(z1, z2)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

> 💡 **核心要点**：整个训练循环中，标签 `_` 从未被使用。模型完全通过对比学习自动学习特征表示。

### Web 应用层 app.py

`app.py` 负责构建 Gradio Web 界面，提供 5 个标签页的交互功能：

| 标签页 | 功能 | 对应回调函数 | 调用的核心方法 |
|:---|:---|:---|:---|
| **1️⃣ 数据准备** | 下载并准备 CIFAR-10 子集 | `prepare_data()` | `trainer.prepare_data()` |
| **2️⃣ 数据增强可视化** | 展示同一图片的多个增强视图 | `show_augmentation()` | `visualizer.plot_augmentation_demo()` |
| **3️⃣ 模型架构** | 展示 SimCLR 架构示意图 | `show_architecture()` | `visualizer.plot_architecture_diagram()` |
| **4️⃣ 自监督预训练** | 运行对比学习训练 | `run_pretrain()` | `trainer.pretrain()` |
| **5️⃣ 评估与对比** | 线性评估 + 对比实验 | `run_linear_eval()` / `run_full_comparison()` | `trainer.linear_evaluation()` / `trainer.get_random_baseline()` |

**全局状态管理**：

```python
# 全局训练器和可视化器实例
trainer = SelfSupervisedTrainer()   # 管理模型和数据的生命周期
visualizer = Visualizer()           # 负责所有可视化图表
```

---

## 快速开始

### 环境要求

- Python >= 3.8
- 支持 CUDA 的 GPU（可选，CPU 也可运行）

### 安装步骤

```bash
# 1. 克隆或下载项目
cd 自监督神经网络

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python app.py
```

启动后，程序会在控制台输出：

```
============================================================
🧠 自监督神经网络学习演示
============================================================
🖥️  设备: CPU (或 CUDA GPU)
📌 请按照界面中 1→2→3→4→5 的顺序操作
============================================================
```

然后在浏览器中打开 **http://127.0.0.1:7860** 即可使用。

### 数据集

本项目使用 **CIFAR-10** 数据集：
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 图片尺寸：32×32 彩色
- 训练集：50,000 张（Demo 默认使用 2,000 张子集）
- 测试集：10,000 张（Demo 默认使用 500 张子集）

> 首次运行会自动下载数据集到 `data/` 目录，如已存在则直接加载。

---

## 使用流程

### 步骤 1：数据准备

1. 设置训练样本数（500~5000，默认 2000）
2. 设置测试样本数（200~2000，默认 500）
3. 点击 **"📥 准备数据"** 按钮

> 更多样本可获得更好的学习效果，但训练时间也会增加。

### 步骤 2：数据增强可视化

点击 **"🎲 生成增强视图演示"** 按钮，观察同一张图片经过不同随机变换后的多个视图。这是对比学习的基础——模型需要学会识别这些不同的视图来自同一张图片。

### 步骤 3：模型架构

点击 **"📐 显示架构图"** 按钮，查看 SimCLR 的完整架构示意图，理解数据如何从输入图片流经编码器和投影头。

### 步骤 4：自监督预训练

1. 调整超参数（或使用默认值）
2. 点击 **"🚀 开始自监督预训练"** 按钮
3. 观察训练损失曲线下降——这意味着模型正在学习区分正/负样本对

### 步骤 5：评估与对比

- **📊 运行线性评估**：在预训练特征上训练线性分类器，查看特征空间的 t-SNE 可视化
- **🔬 运行完整对比实验**：对比自监督特征 vs 随机特征的准确率差异

---

## 超参数说明

| 超参数 | 默认值 | 范围 | 说明 |
|:---|:---:|:---:|:---|
| **训练轮次 (Epochs)** | 10 | 3~50 | 更多轮次可获得更好的特征，但耗时更长 |
| **批次大小 (Batch Size)** | 128 | 32~256 | SimCLR 中更大 batch 通常效果更好（提供更多负样本） |
| **学习率 (Learning Rate)** | 0.001 | 0.0001~0.01 | 控制参数更新步长，配合余弦退火调度器使用 |
| **温度 τ (Temperature)** | 0.5 | 0.1~1.0 | 控制对比损失的分布锐度，较小值更关注困难负样本 |
| **特征维度 (Feature Dim)** | 128 | 固定 | 编码器输出特征向量维度 |
| **投影维度 (Projection Dim)** | 64 | 固定 | 投影头输出维度，对比损失在此空间计算 |

### 优化策略

- **优化器**：Adam (weight_decay=1e-4)
- **学习率调度**：余弦退火 (CosineAnnealingLR)，使学习率从初始值平滑衰减至接近 0

---

## 技术细节与设计决策

### 1. 为什么使用简化 ResNet 而非完整 ResNet-50？

本项目面向教学演示，CIFAR-10 图片尺寸仅 32×32，完整 ResNet-50 过于庞大。简化版 ResNet（3 个残差块，通道数 32→64→128）在保留残差学习思想的同时，大幅降低了计算量，适合 CPU 环境下快速训练。

### 2. 投影头在下游任务中为什么要丢弃？

SimCLR 论文的关键发现：投影头会将编码器特征映射到一个对对比学习最优的空间，在这个过程中可能丢弃对下游任务有用的信息（如颜色、方向等）。因此，下游任务应使用编码器的输出 h，而非投影头的输出 z。

### 3. 数据集子集策略

为了演示效率，默认只使用 CIFAR-10 的子集（训练 2000 张，测试 500 张）。用户可通过滑块调整样本数量。子集通过 `np.random.choice` 随机采样，使用 `torch.utils.data.Subset` 包装。

### 4. 特征提取策略

特征提取时使用标准化变换（不含随机增强），确保同一图片每次产生相同的特征向量。这与训练阶段的随机增强策略不同。

### 5. 线性评估使用 scikit-learn

为简化实现，线性评估阶段使用 scikit-learn 的 `LogisticRegression` 而非 PyTorch 全连接层。这避免了额外的训练循环代码，同时 L-BFGS 优化器可以快速收敛。

---

## 扩展与改进方向

| 方向 | 描述 |
|:---|:---|
| **使用完整数据集** | 将训练样本数增加到 50,000，观察特征质量提升 |
| **更强的编码器** | 替换为标准 ResNet-18/34，提升特征表达能力 |
| **引入 MoCo** | 实现动量对比学习，使用动量编码器和队列机制 |
| **引入 BYOL** | 实现不需要负样本的自监督方法 |
| **掩码自编码器 (MAE)** | 实现基于掩码预测的自监督方法 |
| **多数据集支持** | 添加 STL-10、ImageNet-100 等更大规模数据集 |
| **半监督微调** | 使用少量标签对预训练模型进行微调 |
| **TensorBoard 集成** | 添加更丰富的训练监控功能 |

---

## 参考文献

1. **SimCLR**: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)

2. **MoCo**: He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)

3. **BYOL**: Grill, J.B., et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)

4. **MAE**: He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

5. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

---

*本项目使用 PyTorch + Gradio 构建 | 数据集：CIFAR-10 | 框架：SimCLR*