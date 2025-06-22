import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Router(nn.Module):
    def __init__(self, expert_nums, embed_dim=32, top_k=2, noise_epsilon=1e-2, degradation_channels=3):
        """
        参数说明：
        - expert_nums: 专家数量。
        - embed_dim: 嵌入向量的维度。
        - top_k: 最后选择 top-k 个专家。
        - noise_epsilon: 加噪声时的最小标准差，防止标准差为0。
        - degradation_channels: 退化信息图片的通道数（如RGB图片为3）。
        """
        super().__init__()
        self.expert_nums = expert_nums
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon

        # 1. 处理 task_id 的部分：由于任务编号可能非常大，使用简单的 MLP 将数值映射到 embed_dim 空间
        self.task_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 2. 处理退化信息图片部分：用轻量的卷积网络提取特征，再映射到 embed_dim 维度
        self.degradation_conv = nn.Sequential(
            nn.Conv2d(degradation_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 使用自适应平均池化将特征图压缩为 1x1，相当于全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.degradation_fc = nn.Linear(32, embed_dim)

        # 3. 专家键向量：每个专家用一个可训练向量表示，形状为 [expert_nums, embed_dim]
        self.expert_keys = nn.Parameter(torch.randn(expert_nums, embed_dim))

        # 4. 融合权重设置：希望 task_id 起主导作用，这里设定 alpha=0.7，则退化信息权重为 (1 - 0.7)=0.3
        self.alpha = 0.8

        # 5. 全连接层：分别用于映射到 logits 和计算噪声标准差
        self.fc_gate = nn.Linear(expert_nums, expert_nums)
        self.fc_noise = nn.Linear(expert_nums, expert_nums)

        # 6. 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, task_id, degradation_info, train=True):
        """
        前向传播说明：
        - task_id: Tensor，形状 [batch_size]，每个元素为任务编号（整数）
        - degradation_info: Tensor，形状 [batch_size, degradation_channels, H, W]，退化信息图片
        - train: 布尔值，指示是否为训练模式（训练时加入噪声）
        """
        batch_size = task_id.shape[0]
        # 处理 task_id：转换为浮点数并扩展维度 [batch_size, 1]，然后通过 MLP 得到嵌入
        task_id_float = task_id.float().unsqueeze(1)
        task_embed = self.task_mlp(task_id_float)  # 形状 [batch_size, embed_dim]

        # 处理退化信息图片：先通过卷积网络提取特征，再用全连接层映射到 embed_dim
        degradation_features = self.degradation_conv(degradation_info)  # 形状 [batch_size, 32, 1, 1]
        degradation_features = degradation_features.view(batch_size, -1)  # 展平成 [batch_size, 32]
        degradation_embed = self.degradation_fc(degradation_features)       # 形状 [batch_size, embed_dim]
        # 融合两个输入的嵌入：task_embed 为主导，退化信息嵌入乘以较小比例 (1 - self.alpha)
        combined_embed = task_embed + (1 - self.alpha) * degradation_embed  # 形状 [batch_size, embed_dim]

        # 计算专家与输入之间的相似度：
        # 将 combined_embed 与 expert_keys 作点积，得到相似度得分，形状 [batch_size, expert_nums]
        # 这里使用缩放因子 sqrt(embed_dim) 来稳定数值（类似于 Transformer 中的缩放点积注意力）
        scaling_factor = math.sqrt(self.embed_dim)
        similarity = combined_embed @ self.expert_keys.t() / scaling_factor

        # 用 softmax 将相似度转换为概率分布（专家权重）
        expert_weight = self.softmax(similarity)  # 形状 [batch_size, expert_nums]

        # 计算 logits，并在训练时加入噪声（后续部分保持不变）
        clean_logits = self.fc_gate(expert_weight)
        if train:
            raw_noise_stddev = self.fc_noise(expert_weight)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Top-k 专家选择：
        # 选择每个样本 logits 中最大的 top_k 个专家，并用 softmax 归一化这部分 logits 得到 gate 权重
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_nums), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # 构造与 logits 同尺寸的全0张量，然后将 top_k 的权重放入对应位置
        zeros = torch.zeros_like(logits).to(top_k_gates.dtype)
        try:
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
        except RuntimeError as e:
            print("Scatter 操作失败，错误信息如下：")
            print(e)
            print("zeros.dtype:", zeros.dtype)
            print("top_k_indices.dtype:", top_k_indices.dtype)
            print("top_k_gates.dtype:", top_k_gates.dtype)
            raise

        # 计算专家负载：简单地将所有样本中每个专家的 gate 权重求和
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)
