"""
自监督神经网络演示Demo - Gradio 可视化交互界面

启动方法: python app.py
然后在浏览器中打开显示的URL（通常是 http://127.0.0.1:7860）
"""

import gradio as gr
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from self_supervised_core import SelfSupervisedTrainer, Visualizer

# ============================================================
# 全局状态
# ============================================================
trainer = SelfSupervisedTrainer()
visualizer = Visualizer()

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 回调函数
# ============================================================


def prepare_data(num_train, num_test):
    """准备数据集"""
    msg = trainer.prepare_data(
        num_train_samples=int(num_train), num_test_samples=int(num_test)
    )
    return msg


def show_augmentation():
    """展示数据增强效果"""
    if trainer.train_dataset is None:
        return None, "❌ 请先在 '1. 数据准备' 标签页中准备数据！"
    fig = visualizer.plot_augmentation_demo(trainer.train_dataset)
    return fig, "✅ 数据增强演示已生成。观察同一图片的不同视图 —— 这是自监督学习的基础！"


def show_architecture():
    """展示模型架构图"""
    fig = visualizer.plot_architecture_diagram()
    return fig


def run_pretrain(epochs, batch_size, lr, temperature, progress=gr.Progress()):
    """运行自监督预训练"""
    if trainer.train_dataset is None:
        return None, "❌ 请先准备数据！"

    log_messages = []

    def progress_callback(epoch, total, loss, elapsed):
        progress(epoch / total, desc=f"训练中: Epoch {epoch}/{total}")
        log_messages.append(
            f"[Epoch {epoch:3d}/{total}] 损失: {loss:.4f} | 已用时: {elapsed:.1f}秒"
        )

    history, msg = trainer.pretrain(
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        temperature=float(temperature),
        progress_callback=progress_callback,
    )

    if history is None:
        return None, msg

    fig = visualizer.plot_training_loss(history)
    log_text = "\n".join(log_messages) + "\n\n" + msg
    return fig, log_text


def run_linear_eval():
    """运行线性评估"""
    if trainer.model is None:
        return None, None, "❌ 请先完成自监督预训练！"

    result, msg = trainer.linear_evaluation()
    if result is None:
        return None, None, msg

    train_acc, test_acc, train_features, train_labels, test_features, test_labels = result

    # 生成特征空间可视化
    feature_fig = visualizer.plot_feature_space(
        test_features,
        test_labels,
        title="自监督学习特征空间",
        method="tsne",
    )

    return feature_fig, None, msg


def run_full_comparison():
    """运行完整对比实验"""
    if trainer.model is None:
        return None, None, "❌ 请先完成自监督预训练！"

    # 获取自监督特征
    eval_result, eval_msg = trainer.linear_evaluation()
    if eval_result is None:
        return None, None, eval_msg

    train_acc, test_acc, train_features, train_labels, test_features, test_labels = eval_result

    # 获取随机基线
    random_acc, random_features, random_labels = trainer.get_random_baseline()

    # 生成对比图
    comparison_fig = visualizer.plot_comparison(
        random_features,
        random_labels,
        test_features,
        test_labels,
        random_acc,
        test_acc,
    )

    summary = (
        f"🔬 完整对比实验结果：\n"
        f"{'='*50}\n"
        f"📌 随机初始化编码器 + 线性分类器:  {random_acc*100:.1f}%\n"
        f"📌 自监督预训练编码器 + 线性分类器: {test_acc*100:.1f}%\n"
        f"📌 随机猜测基线:                   10.0%\n"
        f"{'='*50}\n\n"
        f"📈 自监督学习提升: +{(test_acc - random_acc)*100:.1f}% (相对随机初始化)\n\n"
        f"💡 核心发现:\n"
        f"  自监督对比学习在完全不使用任何标签的情况下，\n"
        f"  学到了比随机初始化好得多的特征表示！\n"
        f"  仅用一个简单的线性分类器就能达到 {test_acc*100:.1f}% 的准确率。"
    )

    return comparison_fig, None, summary


# ============================================================
# Gradio 界面构建
# ============================================================


def build_app():
    """构建Gradio应用界面"""

    with gr.Blocks(
        title="🧠 自监督神经网络学习演示",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="emerald",
        ),
    ) as app:

        gr.Markdown(
            """
            # 🧠 自监督神经网络 (Self-Supervised Learning) 交互式学习演示

            ## 什么是自监督学习？

            **自监督学习**是一种**不需要人工标签**的深度学习方法。它通过设计巧妙的**代理任务 (Pretext Task)**，
            让模型从数据本身学习有用的特征表示。

            ### 🎯 本Demo演示的是 SimCLR —— 一种经典的对比学习方法

            **核心思想**：对同一张图片施加不同的随机变换（如裁剪、翻转、颜色变化），得到两个"视图"。
            网络需要学会让**同一图片的不同视图特征相似**，**不同图片的视图特征不同**。

            ### 📋 使用步骤

            按顺序操作以下5个标签页，体验完整的自监督学习流程：

            | 步骤 | 标签页 | 说明 |
            |:---:|:---:|:---|
            | 1️⃣ | 数据准备 | 下载并准备CIFAR-10数据集 |
            | 2️⃣ | 数据增强可视化 | 理解自监督学习的数据增强策略 |
            | 3️⃣ | 模型架构 | 了解SimCLR的网络结构 |
            | 4️⃣ | 自监督预训练 | 运行对比学习训练（不使用标签！） |
            | 5️⃣ | 评估与对比 | 评估学到的特征质量 |
            """
        )

        # ==================== Tab 1: 数据准备 ====================
        with gr.Tab("1️⃣ 数据准备"):
            gr.Markdown(
                """
                ## 📦 数据准备

                我们使用 **CIFAR-10** 数据集，包含10类32×32彩色图片：
                飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

                > 💡 **关键点**：在自监督预训练阶段，我们**完全不使用标签**！
                > 标签仅在最后的评估阶段使用，用来检验学到的特征质量。

                为了加快演示速度，默认使用数据子集。首次运行会自动下载数据集。
                """
            )
            with gr.Row():
                num_train = gr.Slider(
                    500, 5000, 2000, step=500,
                    label="训练样本数",
                    info="更多样本 → 更好的学习效果，但训练更慢",
                )
                num_test = gr.Slider(
                    200, 2000, 500, step=100,
                    label="测试样本数",
                    info="用于评估特征质量",
                )
            prepare_btn = gr.Button("📥 准备数据", variant="primary", size="lg")
            prepare_output = gr.Textbox(label="状态", lines=2)
            prepare_btn.click(
                fn=prepare_data,
                inputs=[num_train, num_test],
                outputs=prepare_output,
            )

        # ==================== Tab 2: 数据增强 ====================
        with gr.Tab("2️⃣ 数据增强可视化"):
            gr.Markdown(
                """
                ## 🎨 数据增强 —— 自监督学习的核心驱动力

                数据增强是自监督对比学习的核心。对同一张图片施加**不同的随机变换**，
                生成多个"视图"。网络需要学会：**这些看起来不同的视图其实来自同一张图片**。

                ### 使用的增强操作：
                - 🔲 **随机裁剪并缩放** - 让网络学习不同尺度和位置的特征
                - ↔️ **随机水平翻转** - 让网络学习方向不变性
                - 🌈 **颜色抖动** - 改变亮度、对比度、饱和度、色调
                - ⬛ **随机灰度化** - 让网络不过度依赖颜色信息

                点击下方按钮查看增强效果：
                """
            )
            aug_btn = gr.Button("🎲 生成增强视图演示", variant="primary", size="lg")
            aug_plot = gr.Plot(label="数据增强效果")
            aug_msg = gr.Textbox(label="说明", lines=2)
            aug_btn.click(fn=show_augmentation, outputs=[aug_plot, aug_msg])

        # ==================== Tab 3: 模型架构 ====================
        with gr.Tab("3️⃣ 模型架构"):
            gr.Markdown(
                """
                ## 🏗️ SimCLR 模型架构

                SimCLR 由两个核心组件构成：

                ### 1. 编码器 f（Feature Encoder）
                - 基于简化版 ResNet
                - 将32×32的图片映射为128维特征向量
                - **这是我们真正想要训练的部分** —— 训练后用于下游任务

                ### 2. 投影头 g（Projection Head）
                - 简单的两层MLP (128 → 128 → 64)
                - 将特征进一步映射到对比学习空间
                - **训练完成后丢弃** —— 仅在训练时使用

                > 🔑 **为什么需要投影头？**
                > SimCLR论文发现：在投影头输出空间计算对比损失效果更好。
                > 投影头可能会丢弃一些对对比学习无用但对下游任务有用的信息，
                > 所以下游任务使用编码器的输出（而非投影头的输出）。
                """
            )
            arch_btn = gr.Button("📐 显示架构图", variant="primary", size="lg")
            arch_plot = gr.Plot(label="SimCLR 架构示意图")
            arch_btn.click(fn=show_architecture, outputs=arch_plot)

            gr.Markdown(
                """
                ### 📊 NT-Xent 对比损失函数

                **归一化温度缩放交叉熵损失** (Normalized Temperature-scaled Cross Entropy Loss)

                给定一个batch中的N张图片，每张生成2个视图，共2N个样本：

                ```
                损失计算过程：
                1. 对每个样本，计算它与所有其他样本的余弦相似度
                2. "正样本对"：同一图片的两个视图 → 希望相似度高
                3. "负样本对"：不同图片的视图 → 希望相似度低
                4. 用温度参数τ缩放相似度，然后计算交叉熵损失

                        exp(sim(z_i, z_j) / τ)
                L = -log ─────────────────────────────
                        Σ_{k≠i} exp(sim(z_i, z_k) / τ)
                ```

                **温度τ的作用**：
                - τ 较小 → 更关注困难负样本（与正样本相似的负样本）
                - τ 较大 → 更均匀地对待所有负样本
                """
            )

        # ==================== Tab 4: 预训练 ====================
        with gr.Tab("4️⃣ 自监督预训练"):
            gr.Markdown(
                """
                ## 🏋️ 自监督预训练

                现在开始训练！请注意：**整个训练过程完全不使用任何标签！**

                模型仅通过对比不同图片的增强视图来学习特征表示。

                ### 超参数说明：
                - **训练轮次**: 更多轮次 → 更好的特征，但耗时更长
                - **批次大小**: SimCLR中更大的batch通常效果更好（更多负样本）
                - **学习率**: 控制参数更新步长
                - **温度**: 对比损失中的温度参数τ
                """
            )
            with gr.Row():
                epochs = gr.Slider(
                    3, 50, 10, step=1,
                    label="训练轮次 (Epochs)",
                    info="建议首次体验用5-10轮",
                )
                batch_size = gr.Slider(
                    32, 256, 128, step=32,
                    label="批次大小 (Batch Size)",
                    info="更大batch = 更多负样本 = 通常更好",
                )
            with gr.Row():
                lr = gr.Slider(
                    0.0001, 0.01, 0.001, step=0.0001,
                    label="学习率 (Learning Rate)",
                )
                temperature = gr.Slider(
                    0.1, 1.0, 0.5, step=0.05,
                    label="温度 τ (Temperature)",
                    info="较小τ → 关注困难样本",
                )

            train_btn = gr.Button("🚀 开始自监督预训练", variant="primary", size="lg")
            train_plot = gr.Plot(label="训练损失曲线")
            train_log = gr.Textbox(label="训练日志", lines=10)

            train_btn.click(
                fn=run_pretrain,
                inputs=[epochs, batch_size, lr, temperature],
                outputs=[train_plot, train_log],
            )

        # ==================== Tab 5: 评估与对比 ====================
        with gr.Tab("5️⃣ 评估与对比"):
            gr.Markdown(
                """
                ## 🔬 评估自监督学习的效果

                ### 线性评估协议 (Linear Evaluation Protocol)

                这是评估自监督学习特征质量的标准方法：
                1. **冻结编码器** —— 不再更新编码器参数
                2. **在编码器输出上训练一个线性分类器** —— 仅训练一层全连接层
                3. **如果准确率高** → 说明编码器学到了好的特征

                > 💡 **为什么用线性分类器？**
                > 如果一个简单的线性分类器就能在学到的特征上取得好成绩，
                > 说明特征空间已经很好地将不同类别分开了。
                > 这证明了自监督学习确实学到了**有语义意义**的特征表示。

                ### 对比实验
                我们对比三种情况：
                1. ❌ **随机初始化编码器** + 线性分类器 → 特征质量差
                2. ✅ **自监督预训练编码器** + 线性分类器 → 特征质量好
                3. 🎲 **随机猜测基线** → 10%（10个类别）
                """
            )

            with gr.Row():
                eval_btn = gr.Button("📊 运行线性评估", variant="primary", size="lg")
                compare_btn = gr.Button("🔬 运行完整对比实验", variant="secondary", size="lg")

            eval_plot = gr.Plot(label="特征空间可视化")
            compare_plot = gr.Plot(label="对比实验结果")
            eval_msg = gr.Textbox(label="评估结果", lines=10)

            eval_btn.click(
                fn=run_linear_eval,
                outputs=[eval_plot, compare_plot, eval_msg],
            )
            compare_btn.click(
                fn=run_full_comparison,
                outputs=[compare_plot, eval_plot, eval_msg],
            )

        # ==================== 知识总结 ====================
        gr.Markdown(
            """
            ---
            ## 📚 自监督学习知识总结

            ### 为什么自监督学习很重要？

            | 对比维度 | 监督学习 | 自监督学习 |
            |:---:|:---:|:---:|
            | **标签需求** | 需要大量人工标注 | ❌ 不需要标签 |
            | **数据利用** | 仅能使用有标签数据 | 可利用海量无标签数据 |
            | **标注成本** | 高（特别是医学等领域） | 低 |
            | **应用场景** | 有充足标签的任务 | 标签稀缺的任务 |

            ### 自监督学习的主要范式

            1. **对比学习 (Contrastive Learning)** ← 本Demo演示
               - SimCLR, MoCo, BYOL, SwAV等
               - 核心：拉近正样本对，推远负样本对

            2. **掩码预测 (Masked Prediction)**
               - MAE (Masked Autoencoder), BEiT
               - 核心：遮住图片的一部分，让模型预测被遮住的内容

            3. **自蒸馏 (Self-Distillation)**
               - DINO, iBOT
               - 核心：学生网络学习教师网络的输出

            ### 推荐进一步学习资源
            - 📄 SimCLR论文: "A Simple Framework for Contrastive Learning of Visual Representations"
            - 📄 MoCo论文: "Momentum Contrast for Unsupervised Visual Representation Learning"
            - 📄 BYOL论文: "Bootstrap Your Own Latent"
            - 📄 MAE论文: "Masked Autoencoders Are Scalable Vision Learners"

            ---
            *本Demo使用 PyTorch + Gradio 构建 | CIFAR-10数据集 | SimCLR框架*
            """
        )

    return app


# ============================================================
# 启动应用
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 自监督神经网络学习演示")
    print("=" * 60)
    print(f"🖥️  设备: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    print("📌 请按照界面中 1→2→3→4→5 的顺序操作")
    print("=" * 60)

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
