import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora.layer import LoraLayer, Linear, Conv2d
from model.utils.Adapter import  Linear_Lora, Conv2d_Lora
from typing import Any


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, embd_dim, H, W]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self):
        """Create a SparseDispatcher."""
        
        self.use_lora = True
        self.use_detach = False

    def set_use_lora(self, flag):
        self.use_lora = flag
        
    def set_use_detach(self, flag):
        self.use_detach = flag
        
    def updata(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # sort in col respectively
    

        # print(sorted_experts, index_sorted_experts) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # print(self._expert_index)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print(self._batch_index)
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # print(self._part_sizes)
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        # print(gates_exp)
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        # print(self._nonzero_gates)
        
    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        # print(self._batch_index)
        # print(self._part_sizes)
        inp_exp = inp[self._batch_index]
        # inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)
    
    def prompt_dispatch(self, prompt):
        prompt_exp = []
        for i in range(len(self._batch_index)):
          prompt_exp.append(prompt[self._batch_index[i]])
        out = []
        left = 0
        for i in range(len(self._part_sizes)):
          right = left + self._part_sizes[i]
          out.append(prompt[left:right])
          left += self._part_sizes[i]
        
        return out

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)
        # print(stitched.shape)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined
    
    # def all_combine(self, expert_out:torch.Tensor):
    #     """
    #     Input:
    #             expert_out: [experts_num, batch_size, <feature_shape>]
    #     output: [batch_size, Feature]
    #     """
    #     flat_expert_out = expert_out.view(expert_out.shape[0], -1)
    #     out = flat_expert_out.mul(self._gates.T)
        
        
        
    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

def _gates_to_load(gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
def _prob_in_top_k(top_k, clean_values, noisy_values, noise_stddev, noisy_top_values):
    from torch.distributions.normal import Normal
    """Helper function to NoisyTopKGating.
    Computes the probability that value is in top k, given different random noise.
    This gives us a way of backpropagating from a loss that balances the number
    of times each expert is in the top k experts per example.
    In the case of no noise, pass in None for noise_stddev, and the result will
    not be differentiable.
    Args:
    clean_values: a `Tensor` of shape [batch, n].
    noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
      normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a `Tensor` of shape [batch, n], or None
    noisy_top_values: a `Tensor` of shape [batch, m].
        "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
    Returns:
    a `Tensor` of shape [batch, n].
    """
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()

    threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + top_k
    threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
    normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob

def noisy_top_k_gating(self,expert_num, top_k, x, train, w_gate, w_noise, noise_epsilon=1e-2):
    """Noisy top-k gating.
      See paper: https://arxiv.org/abs/1701.06538.
      Args:
        x: input Tensor with shape [batch_size, input_size]
        train: a boolean - we only add noise at training time.
        w_gate: [dim, num_expert]
        noise_epsilon: a float
      Returns:
        gates: a Tensor with shape [batch_size, num_experts]
        load: a Tensor with shape [num_experts]
    """
    softplus = nn.Softplus()
    softmax = nn.Softmax(1)
    clean_logits = x @ w_gate.to(x)
    if  train:
        raw_noise_stddev = x @ w_noise.to(x)
        noise_stddev = ((softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    # calculate topk + 1 that will be needed for the noisy gates
    top_logits, top_indices = logits.topk(min(top_k + 1, expert_num), dim=1)
    top_k_logits = top_logits[:, :top_k]
    top_k_indices = top_indices[:, :top_k]
    top_k_gates = softmax(top_k_logits)

    zeros = torch.zeros_like(logits)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    # TODO To help get better performence,may need to add later
    # if train:
    #     load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    # else:
    #     load = self._gates_to_load(gates)
    load = _gates_to_load(gates)
    return gates, load

    
      
      
class NoisyTopKGatingNetwork(nn.Module):
    def __init__(self, input_channels, expert_num, top_k, noise_epsilon=1e-2):
        super(NoisyTopKGatingNetwork, self).__init__()
        self.expert_num = expert_num
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon

        # 局部特征提取层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # 噪声估计器
        self.noise_estimator = nn.Conv2d(256, 1, kernel_size=1)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc_gate = nn.Linear(256, expert_num)
        self.fc_noise = nn.Linear(256, expert_num)

        # Softplus 和 Softmax 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, train=True):
        '''
        	x: [B,C,H,W]
        '''
        # 1. 提取局部特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 2. 噪声估计
        noise_map = self.noise_estimator(x)
        # HACK:噪声融合的方式可以优化？
        x = x + noise_map * x  # 将噪声映射结合到特征中

        # 4. 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 将特征转换为 (B, C'')

        # 5. 计算 clean 和 noisy logits
        clean_logits = self.fc_gate(x)
        if train:
            raw_noise_stddev = self.fc_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 6. 计算 top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 7. 计算专家的负载
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)

# 考虑直接利用修改过的Swinir的shellow_feature层来输入做特征提取
# 同时相比上面直接使用noise_map来做一个输入
class NoisyTopKGatingNetwork_frozen(nn.Module):
    def __init__(self,expert_num, top_k, embed_dim,noise_epsilon=1e-2):
        super(NoisyTopKGatingNetwork_frozen, self).__init__()
        self.expert_num = expert_num
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon

        # 噪声估计器
        self.noise_estimator = nn.Conv2d(1, 1, kernel_size=1)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc_gate = nn.Linear(embed_dim, expert_num)
        self.fc_noise = nn.Linear(embed_dim, expert_num)

        # Softplus 和 Softmax 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)	# 对行化为概率

    def forward(self, x, m,train=True):
        '''
            x: [B,C_embed,H,W] embed feature
            m: [B,1,H,W]	noise_map
        '''

        # 2. 噪声估计
        noise_map = self.noise_estimator(m)
        # HACK:噪声融合的方式可以优化？
        # noise_map * x 即每个通道的矩阵对应位置相乘
        x = x + noise_map * x  # 将噪声映射结合到特征中

        # 4. 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 将特征转换为 (B, C'')

        # 5. 计算 clean 和 noisy logits
        clean_logits = self.fc_gate(x)
        if train:
            raw_noise_stddev = self.fc_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 6. 计算 top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 7. 计算专家的负载
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)
    
    

class SimpleNoisyTopKGatingNetwork(nn.Module):
    def __init__(self, input_channels, expert_num, top_k, noise_epsilon=1e-2):
        super(SimpleNoisyTopKGatingNetwork, self).__init__()
        self.expert_num = expert_num
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon

        # 局部特征提取层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc_gate = nn.Linear(64, expert_num)
        self.fc_noise = nn.Linear(64, expert_num)

        # Softplus 和 Softmax 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, train=True):
        '''
        	x: [B,C,H,W]
        '''
        # 1. 提取局部特征
        x = F.relu(self.conv1(x))
        
        # 4. 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 将特征转换为 (B, C'')

        # 5. 计算 clean 和 noisy logits
        clean_logits = self.fc_gate(x)
        if train:
            raw_noise_stddev = self.fc_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 6. 计算 top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 7. 计算专家的负载
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)


################################################# My Moe_Lora ########################################################
######################################################################################################################
class MoEGate(nn.Module):
    def __init__(self, expert_nums, embed_dim=32, num_heads=4, top_k=2, noise_epsilon=1e-2):
        super(MoEGate, self).__init__()
        self.expert_nums = expert_nums
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        

        # 初始化Transformer中的多头注意力模块
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        # 嵌入层，将 taskID 和 noise_level 映射到查询向量空间
        self.taskID_embed = nn.Embedding(5, embed_dim)  # 假设5个任务
        self.noise_embed = nn.Linear(1, embed_dim)      # 噪声水平嵌入

        # 专家键（key）向量，初始化为一个可训练参数
        self.expert_keys = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        self.alpha = 0.7
        # 全连接层
        self.fc_gate = nn.Linear(embed_dim, expert_nums)
        self.fc_noise = nn.Linear(embed_dim, expert_nums)

        # Softplus 和 Softmax 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, taskID, noise_level, train=True):
        '''
        taskID : tensor([2, 2, 1, 1])
        noise_level: tensor([40.6894, 64.7654, 58.0406, 36.7331])
        
        '''
        batch_size = taskID.shape[0]
        # 将 taskID 和 noise_level 映射到查询向量上
        task_embed = self.taskID_embed(taskID)
        noise_embed = self.noise_embed(noise_level.unsqueeze(-1))  # 添加额外维度

        # 查询向量（query）由任务嵌入和噪声嵌入相加而成
        query = self.alpha * task_embed + (1-self.alpha)*noise_embed

        # 将查询向量调整为多头注意力的输入形状：[1, batch_size, embed_dim]
        query = query.unsqueeze(0)
        keys = self.expert_keys.unsqueeze(1).expand(-1, batch_size, -1)  # 调整专家键的形状：[expert_nums, 1, embed_dim]

        # 使用多头注意力来生成专家分布
        attn_output, attn_weights = self.attention(query, keys, keys)

        # 将注意力权重 `attn_weights` 作为 `gate`，形状：[1, batch_size, expert_nums]
        expert_weight = F.softmax(attn_weights, dim=-1).squeeze(1)
        
         # 5. 计算 clean 和 noisy logits
        clean_logits = self.fc_gate(expert_weight)
        if train:
            raw_noise_stddev = self.fc_noise(expert_weight)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 6. 计算 top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_nums), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 7. 计算专家的负载
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)

class MoEGate_task(nn.Module):
    def __init__(self, expert_nums, embed_dim=32, num_heads=4, top_k=2, noise_epsilon=1e-2):
        super(MoEGate_task, self).__init__()
        self.expert_nums = expert_nums
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        

        # 初始化Transformer中的多头注意力模块
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        # 嵌入层，将 taskID 和 noise_level 映射到查询向量空间
        self.taskID_embed = nn.Embedding(6, embed_dim)  # 假设5个任务

        # 专家键（key）向量，初始化为一个可训练参数
        self.expert_keys = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        self.alpha = 0.7
        # 全连接层
        self.fc_gate = nn.Linear(embed_dim, expert_nums)
        self.fc_noise = nn.Linear(embed_dim, expert_nums)

        # Softplus 和 Softmax 激活函数
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, taskID, train=True):
        '''
        taskID : tensor([2, 2, 1, 1])
        noise_level: tensor([40.6894, 64.7654, 58.0406, 36.7331])
        
        '''
        batch_size = taskID.shape[0]
        # 将 taskID 和 noise_level 映射到查询向量上
        task_embed = self.taskID_embed(taskID)

        # 查询向量（query）由任务嵌入和噪声嵌入相加而成
        query = task_embed.unsqueeze(0)

        # 将查询向量调整为多头注意力的输入形状：[1, batch_size, embed_dim]
        keys = self.expert_keys.unsqueeze(1).expand(-1, batch_size, -1)  # 调整专家键的形状：[expert_nums, 1, embed_dim]

        # 使用多头注意力来生成专家分布
        attn_output, attn_weights = self.attention(query, keys, keys)

        # 将注意力权重 `attn_weights` 作为 `gate`，形状：[1, batch_size, expert_nums]
        expert_weight = F.softmax(attn_weights, dim=-1).squeeze(1)
        
         # 5. 计算 clean 和 noisy logits
        clean_logits = self.fc_gate(expert_weight)
        if train:
            raw_noise_stddev = self.fc_noise(expert_weight)
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # # if train:
        # if taskID[0].item() == 4:
        #     mask = torch.ones_like(logits)  # 创建全1的mask
        #     mask[:, [1, 2]] = 0.21  # 将第二个和第三个专家的mask设为0
        #     # 使用 mask 进行调整
        #     logits = logits * mask
        # 6. 计算 top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.expert_nums), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits).to(top_k_gates.dtype)
        try:
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
        except RuntimeError as e:
            print("Scatter operation failed with the following error:")
            print(e)
            print("zeros.dtype:",zeros.dtype)
            print("top_k_indices.dtype:",top_k_indices.dtype)
            print("top_k_gates.dtype:",top_k_gates.dtype)
            raise
            

        # 7. 计算专家的负载
        load = self._gates_to_load(gates)
        return gates, load
    
    def _gates_to_load(self, gates):
        return gates.sum(0)

class Router(nn.Module):
    def __init__(self, args, accelerator):
        super(Router, self).__init__()
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.device = accelerator.device
        
        self.router_list = nn.ModuleList()
        for _ in range(args.task_num):
            # self.router_list.append(SimpleNoisyTopKGatingNetwork(3,self.num_experts,2))
            self.router_list.append(MoEGate_task(expert_nums=args.num_experts, top_k=args.top_k))
        self.router_list.to(accelerator.device)
        
    def set_train(self):
        self.router_list.train()
        for k,v in self.router_list.named_parameters():
            v.requires_grad_(True)
            
    def forward(self, task_id, bsz, is_train=True):
        task = torch.full((bsz,), task_id).to(device=self.device)
        # get gate
        gates, load = self.router_list[task_id](taskID=task, train=is_train)
        return gates, load
    
    def save_model(self, outf):
        torch.save(self.state_dict(), outf)
        
    def load_checkpoint(self, load_path):
        self.load_state_dict(torch.load(load_path, weights_only=True,map_location="cpu"), strict=True)
        

class Moe_layer(nn.Module):
    def __init__(self, 
                 base_layer,
                 dispatcher:SparseDispatcher, 
                 r, 
                 expert_nums = 5,
                 lora_alpha = 8, 
                 lora_dropout = 0,
                 name=None, 
                 **kwargs):
        super().__init__()
        self.num_experts = expert_nums
        self.expert_linear_lists = nn.ModuleList()
        self.dispatcher = dispatcher
        self.base_layer = base_layer
        self.weight = base_layer.weight
        self.name = name
        
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
            for _ in range(expert_nums):
                self.expert_linear_lists.append(Linear_Lora(r=r,
                                                            in_feature=in_features,
                                                            out_feature=out_features))
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
            kernel_size = base_layer.kernel_size
            stride = base_layer.stride
            padding = base_layer.padding
            for _ in range(expert_nums):
                self.expert_linear_lists.append(Conv2d_Lora(r=r,
                                                            in_features=in_features,
                                                            out_features=out_features,
                                                            kernel_size=kernel_size,
                                                            stride=stride,
                                                            padding=padding))
        else: print(f"[ERROR]:base_layer {name}  is not the expected class,which expected to be nn.Linear or nn.Conv2d, but get {base_layer.__class__}")
        
        
    def forward(self, x: torch.Tensor,  *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)

        if not self.dispatcher.use_lora:
            return result
        
        B = x.shape[0]
        # HACK 魔法数字4是实际的batch_szie, 由于SwinIR会出现B size变化的情况，因此需要这里处理一下
        REAL_BATCH_SIZE = 2
        if (B != REAL_BATCH_SIZE) and isinstance(self.base_layer, nn.Linear):
            resahpe_x = x.repeat(REAL_BATCH_SIZE, 1, 1, 1)
            expert_inputs = self.dispatcher.dispatch(resahpe_x)
            # expert依次求解
            expert_outputs = []
            check_expert_input = []
            for i in range(self.num_experts):
                if(expert_inputs[i].shape[0] != 0):
                    # 激活对应的expert
                    check_expert_input.append(expert_inputs[i].shape) 
                    expert_outputs.append(self.expert_linear_lists[i](expert_inputs[i]))
                # else:
                #     for k,v in self.expert_linear_lists[i].named_parameters():
                #         v.grad = torch.zeros_like(v)
            
            # Combin
            origin_shape = []
            i = 0
            while i < len(expert_outputs):
                if expert_outputs[i].shape[0] == 0 :
                    expert_outputs.pop(i)
                else:
                    if len(origin_shape) == 0:
                        origin_shape = expert_outputs[i].shape
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0],-1)
                    i = i + 1
            lora_out = self.dispatcher.combine(expert_outputs)
            check = lora_out
            new_shape = torch.Size([REAL_BATCH_SIZE, *origin_shape[1:]])
            try:
                lora_out = lora_out.view(new_shape)[0]
            except RuntimeError as e:
                raise RuntimeError(
                                    f"DEBUG SWINIR  \n"
                                    f"$$$$$$$$$$$$$$$$$$$$the more information $$$$$$$$$$$$$$$$$$\n"
                                    f"The layer name is: {self.name}\n"
                                    f"Shape mismatch before view: result shape = {result.shape},\n "
                                    f"lora_out shape = {lora_out.shape} (before view)\n"
                                    f"The resahpe_x shape is: {resahpe_x.shape}\n"
                                    f"The expert input shape is: {check_expert_input}\n"
                                    f"The origin shape is: {origin_shape}\n"
                                    f"The new_shape shape is: {new_shape}\n"
                                    f"The check shape is: {check.shape}"
                                )
            result = result + lora_out
            return result
            
            
        x = x.contiguous()  
        expert_inputs = self.dispatcher.dispatch(x)
        # expert依次求解
        expert_outputs = []
        check_expert_input = []
        for i in range(self.num_experts):
            if(expert_inputs[i].shape[0] != 0):
                # 激活对应的expert
                check_expert_input.append(expert_inputs[i].shape) 
                expert_outputs.append(self.expert_linear_lists[i](expert_inputs[i]))
            # else:
            #     for k,v in self.expert_linear_lists[i].named_parameters():
            #         v.grad = torch.zeros_like(v)
        
        # Combin
        origin_shape = []
        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0 :
                expert_outputs.pop(i)
            else:
                if len(origin_shape) == 0:
                    origin_shape = expert_outputs[i].shape
                try:
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0],-1)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"View operation failed with error: {e}\n"
                        f"$$$$$$$$$$$$$$$$$$$$the more information $$$$$$$$$$$$$$$$$$\n"
                        f"The layer name is: {self.name}\n"
                        f"Shape mismatch before view: result shape = {result.shape},\n "
                        f"The input shape is: {x.shape}\n"
                        f"The expert input shape is: {check_expert_input}\n"
                        f"x .is_contiguous():{x.is_contiguous()}\n"
                        f"expert_outputs[i].is_contiguous(): {expert_outputs[i].is_contiguous()}"
                    )
                i = i + 1
        lora_out = self.dispatcher.combine(expert_outputs)
        check = lora_out
        new_shape = torch.Size([B, *origin_shape[1:]])
        try:
            lora_out = lora_out.view(new_shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"View operation failed with error: {e}\n"
                f"$$$$$$$$$$$$$$$$$$$$the more information $$$$$$$$$$$$$$$$$$\n"
                f"The layer name is: {self.name}\n"
                f"Shape mismatch before view: result shape = {result.shape},\n "
                f"lora_out shape = {lora_out.shape} (before view)\n"
                f"The input shape is: {x.shape}\n"
                f"The expert input shape is: {check_expert_input}\n"
                f"The origin shape is: {origin_shape}\n"
                f"The check shape is: {check.shape}"
            )
        # lora_out = lora_out.view(new_shape)
        # 假设 result 和 lora_out 已经定义并计算出来
        # if squezz_flag:
        #     tmp0, tmp1 = lora_out.shape[:2]
        #     lora_out = lora_out.view(lora_out.shape[0]*lora_out.shape[1], *lora_out.shape[2:])
        if result.shape != lora_out.shape:
            # print(f"Shape mismatch: result shape = {result.shape}, lora_out shape = {lora_out.shape}, the layer name is: {self.name}")
            raise ValueError(
                            f"The shapes of result and lora_out do not match.\n"
                            f"$$$$$$$$$$$$$$$$$$$$the more information $$$$$$$$$$$$$$$$$$"
                            f"the layer name is: {self.name} \n"
                            f"Shape mismatch: result shape = {result.shape}, lora_out shape = {lora_out.shape}, \n"
                            f"the input shape is :{x.shape}\n"
                            f"the expert inpit shape is :{check_expert_input} \n"
                            f"the origin shape is :{origin_shape}\n"
                            f"the check shape is :{check.shape}"
                        )
        result = result + lora_out
        return result

def check_class(module, old_class):
    same_flag = False
    if isinstance(old_class, list):
        for _old_class in old_class:
            if isinstance(module, _old_class):
                same_flag = True
                break
    else:
        same_flag = isinstance(module, old_class)
    
    if not same_flag:
        print(f"[ERROR]:base_layer is not the expected class,which expected to be nn.Linear or nn.Conv2d, but get {module.__class__}")
    return same_flag
    

def replace_layers(model, old_class, new_class, **kwargs):
    replaced = False  # 用于检查是否至少有一层被替换     
    for name, module in model.named_children():
        if check_class(module, old_class):
            # 用新的类进行替换
            setattr(model, name, new_class(base_layer=module, **kwargs))
            # 检查替换是否成功
            if check_class(getattr(model, name), new_class):
                print(f"Replaced {name} with {new_class.__name__} successfully!")
                replaced = True
            else:
                print(f"Failed to replace {name} with {new_class.__name__}")
        else:
            # 递归替换子模块
            replace_layers(module, old_class, new_class, **kwargs)
    if not replaced:
        print(f"No {old_class.__name__} layers were found to replace in the model.")


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split('.')
    for attr in attrs:
        if isinstance(obj, nn.ModuleList):
            obj = obj[int(attr)]  # 如果是 ModuleList，转为列表索引
        else:
            obj = getattr(obj, attr)  # 否则直接用 getattr
    return obj


def replace_layers_byname(model, attr_path, old_class, new_class, **kwargs):
    # 使用 get_nested_attr 获取目标层
    target_layer = get_nested_attr(model, attr_path)

    # 确保目标层是预期的 old_class 类型
    if not check_class(target_layer, old_class):
        return

    # 创建新的层
    new_layer = new_class(base_layer=target_layer, **kwargs)

    # 使用 setattr 将新的层替换到模型中
    current_obj = model
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:  # 遍历到倒数第二个属性
        if isinstance(current_obj, torch.nn.ModuleList):
            current_obj = current_obj[int(attr)]  # 列表索引
        else:
            current_obj = getattr(current_obj, attr)  # 直接获取属性

    # 最后一个属性是要替换的目标属性
    final_attr = attrs[-1]
    setattr(current_obj, final_attr, new_layer)

    # 检查替换是否成功
    if isinstance(getattr(current_obj, final_attr), new_class):
        print(f"Replaced {attr_path} with {new_class.__name__} successfully!")
    else:
        print(f"Failed to replace {attr_path} with {new_class.__name__}")

# replace_layers(model, [nn.Conv2d, nn.Linear], Moe_layer, dispatcher=dispatcher, r=4, expert_nums=5,lora_alpha=8)


def getModelSize(model):
    param_size = 0
    param_sum = 0
    
    # 计算可训练参数的大小
    for param in model.parameters():
        if param.requires_grad:  # 只考虑可训练参数
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    # 计算所有缓冲区的大小
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # 转换为MB
    
    print('模型总大小为：{:.3f}MB'.format(all_size))
    print('可训练参数大小为：{:.3f}MB'.format(param_size / 1024 / 1024))
    print('可训练参数数量：{}'.format(param_sum))
    print('缓冲区大小为：{:.3f}MB'.format(buffer_size / 1024 / 1024))
    print('缓冲区数量：{}'.format(buffer_sum))
    
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


 
if __name__ == '__main__':
    print("ok")
    gate = torch.Tensor(
        [[0.0848, 0., 0., 1.4639, 0.],
        [0.0839, 0., 0.8728, 0., 0.],
        [0., 1.3027, 0., 0., 0.2611],
        [0.7257, 0., 0., 0.8780, 0.],
        [0., 0., 0.7253, 0.9025, 0.],
        [0.4828, 0., 0.4828, 0., 0.],
        [0., 0., 0., 0.3483, 0.3472],
        [0., 0.4582, 0., 0.0086, 0.]])
    input = torch.randn(8,4,5,5)
    model = SparseDispatcher(5,gate)
    output = model.dispatch(input)

    # 查看 tuple 的类型和长度
    # print(output)  # 确认是 tuple 类型
    print(len(output))   # 查看 tuple 的长度
    out = []
    # 查看 tuple 中每个张量的形状
    for i, tensor in enumerate(output):
        print(f"Shape of tensor {i}: {tensor.size()}")
        out.append(tensor.view(tensor.shape[0],-1))
    
    # print(out)
    # x = torch.randn(2,3,128,128)
    # getgate = NoisyTopKGatingNetwork(3,5,2)
    # gate, load = getgate(x)
    # print(gate)
    # print(load)