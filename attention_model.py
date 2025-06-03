import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAgentAttention(nn.Module):
    def __init__(self, dim, num_agents=16, num_heads=8):  # 初始化，dim为输入特征的维度，num_agents为代理token的数量，num_heads为多头注意力头数
        super(MultiAgentAttention, self).__init__()
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.dim = dim
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.agent_query = nn.Linear(dim, dim)
        self.agent_key = nn.Linear(dim, dim)
        self.agent_value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Linear projections
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # 计算Q矩阵
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Pooling to get agent tokens
        A = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_agents).transpose(1, 2)
        A_Q = self.agent_query(A).view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        A_K = self.agent_key(A).view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        A_V = self.agent_value(A).view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)

        # Agent aggregation
        attn_agent = (Q @ A_K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_agent = attn_agent.softmax(dim=-1)
        aggregated_agents = (attn_agent @ A_V)

        # Agent broadcasting
        attn = (aggregated_agents @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, C)

        return self.out(out)


class AgentAttention(nn.Module):
    def __init__(self, dim, num_agents=16):
        super(AgentAttention, self).__init__()
        self.num_agents = num_agents
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.agent_query = nn.Linear(dim, dim)
        self.agent_key = nn.Linear(dim, dim)
        self.agent_value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # N, C = x.shape

        # 生成Q，K，V矩阵，矩阵形状与x相同
        Q = self.query(x)  # (B, N, C)
        K = self.key(x)
        V = self.value(x)

        # 计算A矩阵
        A = F.adaptive_avg_pool1d(Q.transpose(1, 2), self.num_agents).transpose(1, 2)  # (B, num_agents, C)

        # Q矩阵与A矩阵相乘，进行Softmax注意力计算
        attn_agent = torch.matmul(Q, A.transpose(-2, -1)) * (C ** -0.5)  # (B, N, num_agents)
        agent_tokens = attn_agent.softmax(dim=-1)  # (B, N, num_agents) Softmax是对某一维度还是对整个batch？

        # AKV矩阵进行Softmax注意力计算
        agent_features = torch.matmul(A, K.transpose(-2, -1)) * (C ** -0.5)  # (B, num_agents, N)
        agent_features = agent_features.softmax(dim=-1)  # (B, num_agents, N)
        agent_features = torch.matmul(agent_features, V)  # (B, num_agents, C)

        # 得到agent注意力
        agent_attn = torch.matmul(agent_tokens, agent_features)  # (B, N, C)
        return self.out(agent_attn)


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        QK = torch.matmul(Q, K.transpose(-2, -1)) * (C ** -0.5)
        QK = QK.softmax(dim=-1)
        QKV = torch.matmul(QK, V)
        return self.out(QKV)


# 示例用法
batch_size = 8
num_tokens = 128
dim = 512

x = torch.randn(batch_size, num_tokens, dim)
agent_attention = Attention(dim)
output = agent_attention(x)
# print(output.shape)  # 输出形状: (batch_size, num_tokens, dim)
