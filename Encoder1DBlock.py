from torch import nn
from MlpBlock import MlpBlock
from torch import Tensor
from MultiHeadAttention import MultiHeadAttention
"""
class Encoder1DBlock(nn.Module):

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):

    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y
"""
class Encoder1DBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, 
                 embed_dim, # 编码维度
                 mlp_hidden_dim, # 隐藏层维度
                 num_heads, # 多头注意力的头数目
                 dropout_rate = 0.,  
                 attention_dropout_rate = 0.):
        
        super(Encoder1DBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            key_size = embed_dim,
            query_size = embed_dim,
            value_size = embed_dim,
            num_hiddens = embed_dim,
            num_heads = num_heads,
            dropout = attention_dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MlpBlock(in_features = embed_dim, 
                            hidden_features = mlp_hidden_dim, 
                            dropout_rate = dropout_rate)
    

    def forward(self, inputs:Tensor):
        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = self.norm(inputs)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + inputs
        
        # MLP block.
        y = self.norm(x)
        y = self.mlp(x)

        return x + y
