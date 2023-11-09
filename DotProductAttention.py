from torch import nn
from torch import Tensor
import torch
import math

# 缩放点积注意力
# a(q, k) = q @ k.T / sqrt(d)
class DotProductAttention(nn.Module):
    
    def __init__(self, dropout:nn.Dropout, **kwargs) -> None:
        super(DotProductAttention, self).__init__(**kwargs) # 初始父类nn.Module
        self.dropout = nn.Dropout(dropout)

    def forward(self, query:Tensor, key:Tensor, value:Tensor, valid_lens = None):
        # 计算缩放点积注意力
        # a(q, k) = q @ k.T / sqrt(d)
        d = query.shape[-1]
        # q @ k.T
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.matmul(self.dropout(self.attention_weights), value)
    
    # 超出有效长度的x部分会置为0
    def masked_softmax(self, x:Tensor, valid_lens:Tensor) -> Tensor:
        if valid_lens is None: # None就直接返回
            return torch.softmax(x, dim=-1)
        else: 
            shape = x.shape 
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1]) 
                    # valid_lens一维的时候把按照shape[1]重复
                    # valid_lens = [1, 2], shape[1] = 2 ==> [1, 1, 2, 2]
            else:
                valid_lens = valid_lens.reshape(-1)

            X = x.reshape(-1, x.shape[-1])
            maxlen = X.shape[1]
            mask = torch.arange((maxlen), dtype=torch.float32,
                                device = X.device)[None, :] < valid_lens[:, None] 
            X[~mask] = -1e6 # 一个较小值的softmax就为0
            return torch.softmax(X.reshape(shape), dim=-1)