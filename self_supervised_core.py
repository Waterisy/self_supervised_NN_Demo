"""
自监督神经网络演示Demo - 核心模型与训练逻辑

本模块实现了基于SimCLR框架的自监督对比学习，包含：
1. 数据增强管道 (SimCLR风格的数据增强)
2. 编码器网络 (基于简化的ResNet)
3. 投影头 (Projection Head)
4. NT-Xent对比损失函数
5. 自监督预训练 + 线性评估（下游任务）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import io
from PIL import Image
import time
import os

# ============================================================
# 第一部分：数据增强 —— 自监督学习的核心驱动力
# ============================================================

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SimCLRAugmentation:
    """
    SimCLR 风格的数据增强管道。

    自监督学习的核心思想：
    对同一张图片施加不同的随机变换，得到两个"视图"(view)。
    网络需要学会：同一张图片的两个视图应该有相似的特征表示，
    而不同图片的视图应该有不同的特征表示。

    增强操作包括：
    - 随机裁剪并缩放
    - 随机水平翻转
    - 随机颜色抖动（亮度、对比度、饱和度、色调）
    - 随机灰度化
    """

    def __init__(self, img_size=32):
        # 颜色抖动变换
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

    def __call__(self, x):
        """对同一张图片生成两个不同的增强视图"""
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class ContrastiveDataset(Dataset):
    """
    对比学习数据集包装器。

    将普通数据集转换为对比学习所需的格式：
    每次取样返回同一图片的两个不同增强视图。
    """

    def __init__(self, base_dataset, augmentation):
        self.base_dataset = base_dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        view1, view2 = self.augmentation(img)
        return view1, view2, label


# ============================================================
# 第二部分：模型架构 —— 编码器 + 投影头
# ============================================================


class SimpleResBlock(nn.Module):
    """简化的残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接（当维度不匹配时需要投影）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接：让梯度更容易流动
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    特征编码器 (f)

    将输入图片映射到一个低维特征空间。
    基于简化版ResNet，输出128维的特征向量。
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = SimpleResBlock(32, 32, stride=1)
        self.layer2 = SimpleResBlock(32, 64, stride=2)
        self.layer3 = SimpleResBlock(64, 128, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        features = self.fc(out)
        return features


class ProjectionHead(nn.Module):
    """
    投影头 (g)

    将编码器的特征进一步映射到一个对比学习的空间。
    关键发现（SimCLR论文）：在投影头的输出空间计算对比损失，
    比直接在编码器输出空间计算效果更好。
    但在下游任务中，我们使用编码器的输出而不是投影头的输出。

    结构：MLP (Linear -> ReLU -> Linear)
    """

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    完整的SimCLR模型 = 编码器(f) + 投影头(g)

    前向流程：
    输入图片 → 编码器 f → 特征表示 h → 投影头 g → 投影 z
                                ↑                        ↑
                        (下游任务使用)            (对比损失使用)
    """

    def __init__(self, feature_dim=128, projection_dim=64):
        super().__init__()
        self.encoder = Encoder(feature_dim=feature_dim)
        self.projection_head = ProjectionHead(
            input_dim=feature_dim, output_dim=projection_dim
        )

    def forward(self, x):
        features = self.encoder(x)  # h = f(x)
        projections = self.projection_head(features)  # z = g(h)
        return features, projections


# ============================================================
# 第三部分：对比损失函数 NT-Xent
# ============================================================


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    归一化温度缩放交叉熵损失

    核心思想：
    给定一个batch中N张图片，每张生成2个视图，共2N个样本。
    对于每个样本，它的"正样本"是同一张图片的另一个视图，
    其余2(N-1)个样本都是"负样本"。

    目标：最大化正样本对的相似度，最小化负样本对的相似度。

    temperature参数控制分布的"锐利程度"：
    - 较小的temperature → 更集中于困难负样本
    - 较大的temperature → 更均匀地对待所有负样本
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: 第一组视图的投影 [batch_size, projection_dim]
        z_j: 第二组视图的投影 [batch_size, projection_dim]
        """
        batch_size = z_i.shape[0]

        # L2归一化，使向量都在单位超球面上
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 拼接所有投影 [2*batch_size, projection_dim]
        z = torch.cat([z_i, z_j], dim=0)

        # 计算所有样本对之间的余弦相似度
        # sim[a][b] = cos(z_a, z_b) / temperature
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # 创建标签：正样本对的位置
        # 对于样本i，它的正样本是样本i+batch_size（反之亦然）
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size),
                torch.arange(0, batch_size),
            ]
        ).to(z.device)

        # 移除对角线（自身与自身的相似度），因为我们不希望样本与自己对比
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -1e9)

        # 使用交叉熵损失：将正样本对视为正确类别
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ============================================================
# 第四部分：训练与评估
# ============================================================


class SelfSupervisedTrainer:
    """
    自监督学习训练器

    包含完整的训练流程：
    1. 自监督预训练（不使用标签）
    2. 特征提取
    3. 线性评估（用少量标签评估特征质量）
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.training_history = {
            "contrastive_loss": [],
            "epoch": [],
            "linear_eval_acc": [],
        }

    def prepare_data(self, num_train_samples=2000, num_test_samples=500):
        """
        准备CIFAR-10数据集

        为了演示效率，默认使用子集。
        CIFAR-10包含10类32x32彩色图片。
        """
        # 下载CIFAR-10（如果不存在）
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )

        full_train = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True
        )
        full_test = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True
        )

        # 使用子集以加快演示速度
        train_indices = np.random.choice(
            len(full_train), num_train_samples, replace=False
        )
        test_indices = np.random.choice(
            len(full_test), num_test_samples, replace=False
        )

        self.train_dataset = Subset(full_train, train_indices)
        self.test_dataset = Subset(full_test, test_indices)

        # 记录类别名称
        self.class_names = [
            "飞机", "汽车", "鸟", "猫", "鹿",
            "狗", "青蛙", "马", "船", "卡车",
        ]

        return f"✅ 数据准备完成！训练集: {num_train_samples} 张, 测试集: {num_test_samples} 张"

    def pretrain(
        self,
        epochs=10,
        batch_size=128,
        lr=0.001,
        temperature=0.5,
        progress_callback=None,
    ):
        """
        自监督预训练阶段

        关键点：整个训练过程完全不使用标签！
        模型通过对比学习自动学习有意义的特征表示。
        """
        if self.train_dataset is None:
            return None, "❌ 请先准备数据！"

        # 初始化模型
        self.model = SimCLRModel(feature_dim=128, projection_dim=64).to(
            self.device
        )

        # 创建对比学习数据集
        augmentation = SimCLRAugmentation(img_size=32)
        contrastive_dataset = ContrastiveDataset(self.train_dataset, augmentation)
        dataloader = DataLoader(
            contrastive_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 损失函数和优化器
        criterion = NTXentLoss(temperature=temperature)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 清空历史
        self.training_history = {
            "contrastive_loss": [],
            "epoch": [],
            "linear_eval_acc": [],
        }

        self.model.train()
        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for view1, view2, _ in dataloader:  # 注意：标签 _ 完全没有使用！
                view1, view2 = view1.to(self.device), view2.to(self.device)

                # 前向传播
                _, z1 = self.model(view1)  # 视图1的投影
                _, z2 = self.model(view2)  # 视图2的投影

                # 计算对比损失
                loss = criterion(z1, z2)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            self.training_history["contrastive_loss"].append(avg_loss)
            self.training_history["epoch"].append(epoch + 1)

            if progress_callback:
                progress_callback(
                    epoch + 1,
                    epochs,
                    avg_loss,
                    time.time() - total_start_time,
                )

        total_time = time.time() - total_start_time
        return (
            self.training_history,
            f"✅ 预训练完成！共 {epochs} 轮, 最终损失: {avg_loss:.4f}, "
            f"耗时: {total_time:.1f}秒",
        )

    def extract_features(self, dataset, max_samples=1000):
        """
        使用训练好的编码器提取特征

        注意：这里只使用编码器(f)的输出，不使用投影头(g)的输出。
        因为编码器的输出包含更多通用信息，更适合下游任务。
        """
        if self.model is None:
            return None, None

        self.model.eval()
        features_list = []
        labels_list = []

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

        count = 0
        with torch.no_grad():
            for i in range(len(dataset)):
                if count >= max_samples:
                    break
                img, label = dataset[i]
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                feature, _ = self.model(img_tensor)
                features_list.append(feature.cpu().numpy())
                labels_list.append(label)
                count += 1

        features = np.concatenate(features_list, axis=0)
        labels = np.array(labels_list)
        return features, labels

    def linear_evaluation(self, num_train=500, num_test=300):
        """
        线性评估协议

        这是评估自监督学习特征质量的标准方法：
        1. 冻结编码器（不再更新参数）
        2. 在编码器输出上训练一个简单的线性分类器
        3. 如果线性分类器就能达到不错的准确率，
           说明编码器学到了好的特征表示

        好的自监督特征 → 线性分类器也能工作良好
        差的特征 → 线性分类器无法分类
        """
        if self.model is None:
            return None, "❌ 请先完成预训练！"

        # 提取特征
        train_features, train_labels = self.extract_features(
            self.train_dataset, max_samples=num_train
        )
        test_features, test_labels = self.extract_features(
            self.test_dataset, max_samples=num_test
        )

        if train_features is None:
            return None, "❌ 特征提取失败！"

        # 训练线性分类器（逻辑回归）
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(train_features, train_labels)

        # 评估
        train_acc = accuracy_score(
            train_labels, classifier.predict(train_features)
        )
        test_acc = accuracy_score(
            test_labels, classifier.predict(test_features)
        )

        self.training_history["linear_eval_acc"].append(test_acc)

        result = (
            f"📊 线性评估结果：\n"
            f"   训练集准确率: {train_acc*100:.1f}%\n"
            f"   测试集准确率: {test_acc*100:.1f}%\n\n"
            f"💡 解读: 如果仅用线性分类器就能达到较高准确率，\n"
            f"   说明自监督学习已经学到了有意义的特征表示！\n"
            f"   (CIFAR-10随机猜测准确率为10%)"
        )

        return (train_acc, test_acc, train_features, train_labels, test_features, test_labels), result

    def get_random_baseline(self, num_train=500, num_test=300):
        """
        随机初始化基线

        使用未经训练的随机编码器提取特征，作为对比基线。
        """
        random_model = SimCLRModel(feature_dim=128, projection_dim=64).to(self.device)
        random_model.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )

        features_list, labels_list = [], []
        with torch.no_grad():
            for i in range(min(num_train, len(self.train_dataset))):
                img, label = self.train_dataset[i]
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                feature, _ = random_model(img_tensor)
                features_list.append(feature.cpu().numpy())
                labels_list.append(label)

        train_features = np.concatenate(features_list, axis=0)
        train_labels = np.array(labels_list)

        features_list, labels_list = [], []
        with torch.no_grad():
            for i in range(min(num_test, len(self.test_dataset))):
                img, label = self.test_dataset[i]
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                feature, _ = random_model(img_tensor)
                features_list.append(feature.cpu().numpy())
                labels_list.append(label)

        test_features = np.concatenate(features_list, axis=0)
        test_labels = np.array(labels_list)

        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(train_features, train_labels)
        random_acc = accuracy_score(test_labels, classifier.predict(test_features))

        return random_acc, train_features, train_labels


# ============================================================
# 第五部分：可视化工具
# ============================================================


class Visualizer:
    """可视化工具集，用于直观展示自监督学习的各个方面"""

    def __init__(self, class_names=None):
        self.class_names = class_names or [
            "飞机", "汽车", "鸟", "猫", "鹿",
            "狗", "青蛙", "马", "船", "卡车",
        ]

    def plot_augmentation_demo(self, dataset, num_images=4, num_views=4):
        """
        可视化数据增强效果

        展示同一张图片经过不同随机变换后的样子。
        这些不同的"视图"是自监督学习的输入。
        """
        fig, axes = plt.subplots(
            num_images,
            num_views + 1,
            figsize=(3 * (num_views + 1), 3 * num_images),
        )
        fig.suptitle(
            "数据增强演示：同一图片的多个不同视图\n"
            "(自监督学习让网络学会：这些视图来自同一张图片)",
            fontsize=14,
            fontweight="bold",
        )

        augmentation = SimCLRAugmentation(img_size=32)

        # 反归一化函数
        inv_normalize = transforms.Normalize(
            mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010],
        )

        indices = np.random.choice(len(dataset), num_images, replace=False)
        for i, idx in enumerate(indices):
            img, label = dataset[idx]

            # 显示原图
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"原图\n({self.class_names[label]})", fontsize=10)
            axes[i, 0].axis("off")

            # 显示增强视图
            for j in range(num_views):
                view, _ = augmentation(img)
                view = inv_normalize(view)
                view = view.permute(1, 2, 0).numpy()
                view = np.clip(view, 0, 1)
                axes[i, j + 1].imshow(view)
                axes[i, j + 1].set_title(f"视图 {j+1}", fontsize=10)
                axes[i, j + 1].axis("off")

        plt.tight_layout()
        return fig

    def plot_training_loss(self, history):
        """
        绘制训练损失曲线

        对比损失下降 → 模型逐渐学会区分正样本对和负样本对
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        epochs = history["epoch"]
        losses = history["contrastive_loss"]

        ax.plot(epochs, losses, "b-o", linewidth=2, markersize=6, label="NT-Xent 对比损失")
        ax.fill_between(epochs, losses, alpha=0.1, color="blue")
        ax.set_xlabel("训练轮次 (Epoch)", fontsize=12)
        ax.set_ylabel("对比损失", fontsize=12)
        ax.set_title(
            "自监督预训练：对比损失变化曲线\n"
            "(损失下降 → 模型学会了区分相似与不同的图片)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_feature_space(
        self,
        features,
        labels,
        title="特征空间可视化",
        method="tsne",
    ):
        """
        将高维特征降到2D进行可视化

        好的特征 → 同类样本聚集，不同类样本分开
        差的特征 → 所有样本混在一起
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)

        features_2d = reducer.fit_transform(features)

        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.6,
            s=20,
        )

        # 添加图例
        legend_elements = []
        for i in range(10):
            mask = labels == i
            if mask.any():
                legend_elements.append(
                    plt.scatter([], [], c=[plt.cm.tab10(i / 10)], label=self.class_names[i], s=40)
                )
        ax.legend(handles=legend_elements, loc="best", fontsize=9, ncol=2)

        ax.set_title(
            f"{title}\n({method.upper()} 降维可视化)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("维度 1", fontsize=11)
        ax.set_ylabel("维度 2", fontsize=11)

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        random_features,
        random_labels,
        trained_features,
        trained_labels,
        random_acc,
        trained_acc,
    ):
        """
        对比随机特征 vs 自监督特征

        直观展示自监督学习的效果
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # --- 子图1: 随机特征 ---
        reducer1 = TSNE(n_components=2, random_state=42, perplexity=30)
        random_2d = reducer1.fit_transform(random_features[:500])
        axes[0].scatter(
            random_2d[:, 0],
            random_2d[:, 1],
            c=random_labels[:500],
            cmap="tab10",
            alpha=0.6,
            s=15,
        )
        axes[0].set_title(
            f"❌ 随机初始化特征\n(线性评估准确率: {random_acc*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("维度 1")
        axes[0].set_ylabel("维度 2")

        # --- 子图2: 自监督特征 ---
        reducer2 = TSNE(n_components=2, random_state=42, perplexity=30)
        trained_2d = reducer2.fit_transform(trained_features[:500])
        axes[1].scatter(
            trained_2d[:, 0],
            trained_2d[:, 1],
            c=trained_labels[:500],
            cmap="tab10",
            alpha=0.6,
            s=15,
        )
        axes[1].set_title(
            f"✅ 自监督学习特征\n(线性评估准确率: {trained_acc*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("维度 1")
        axes[1].set_ylabel("维度 2")

        # --- 子图3: 准确率对比柱状图 ---
        methods = ["随机初始化", "自监督学习", "随机猜测"]
        accs = [random_acc * 100, trained_acc * 100, 10.0]
        colors = ["#ff6b6b", "#51cf66", "#868e96"]

        bars = axes[2].bar(methods, accs, color=colors, edgecolor="black", linewidth=0.5)
        axes[2].set_ylabel("测试准确率 (%)", fontsize=11)
        axes[2].set_title(
            "线性评估准确率对比\n(更高 = 特征质量更好)",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].set_ylim(0, 100)

        # 在柱子上标注数值
        for bar, acc in zip(bars, accs):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        plt.tight_layout()
        return fig

    def plot_architecture_diagram(self):
        """绘制SimCLR架构示意图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(
            "SimCLR 自监督对比学习框架",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # 绘制方块
        def draw_box(x, y, w, h, text, color, fontsize=10):
            rect = plt.Rectangle(
                (x, y), w, h, facecolor=color, edgecolor="black",
                linewidth=1.5, alpha=0.8, zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=3,
            )

        # 绘制箭头
        def draw_arrow(x1, y1, x2, y2, text=""):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"),
                zorder=1,
            )
            if text:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mx, my + 0.3, text, ha="center", fontsize=9, style="italic")

        # 输入图片
        draw_box(0.5, 4, 2, 2, "输入图片\nx", "#a8dadc", fontsize=11)

        # 两个增强视图
        draw_box(4, 7, 2, 1.5, "增强视图 1\nt(x)", "#ffd6a5")
        draw_box(4, 1.5, 2, 1.5, "增强视图 2\nt'(x)", "#ffd6a5")

        # 编码器
        draw_box(7.5, 7, 2, 1.5, "编码器 f\n(共享权重)", "#b5e48c")
        draw_box(7.5, 1.5, 2, 1.5, "编码器 f\n(共享权重)", "#b5e48c")

        # 投影头
        draw_box(11, 7, 2, 1.5, "投影头 g\n→ z_i", "#ddb4f6")
        draw_box(11, 1.5, 2, 1.5, "投影头 g\n→ z_j", "#ddb4f6")

        # 对比损失
        draw_box(10.5, 4.2, 3, 1.6, "NT-Xent\n对比损失", "#ffadad", fontsize=11)

        # 箭头连接
        draw_arrow(2.5, 5.5, 4, 7.75, "随机增强 t")
        draw_arrow(2.5, 4.5, 4, 2.25, "随机增强 t'")
        draw_arrow(6, 7.75, 7.5, 7.75, "")
        draw_arrow(6, 2.25, 7.5, 2.25, "")
        draw_arrow(9.5, 7.75, 11, 7.75, "h → z")
        draw_arrow(9.5, 2.25, 11, 2.25, "h → z")
        draw_arrow(12, 7, 12, 5.8, "")
        draw_arrow(12, 3, 12, 4.2, "")

        # 标注
        ax.text(
            7, 0.3,
            "核心思想: 同一图片的不同增强视图(正样本对)的特征应该相似，\n"
            "不同图片的增强视图(负样本对)的特征应该不同。\n"
            "整个过程完全不需要人工标签！",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"),
        )

        plt.tight_layout()
        return fig
