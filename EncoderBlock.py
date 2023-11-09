from torch import nn
from torch import Tensor
from AddPositionEmbs import AddPositionEmbs
from Encoder1DBlock import Encoder1DBlock
"""
Transformer Model Encoder for sequence to sequence translation.
"""
"""
class Encoder(nn.Module):
 

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded
"""
class Encoder(nn.Module):

    def __init__(self, 
                embed_dim,
                num_layers, 
                mlp_hidden_dim, 
                num_heads, 
                dropout_rate = 0.,
                attention_dropout_rate = 0.,
                add_position_embedding = True,
                num_patches = None) -> None:
        super(Encoder, self).__init__()

        self.add_position_embedding = AddPositionEmbs(num_patches, embed_dim) if add_position_embedding else nn.Identity()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)
        self.encoder_layers = nn.Sequential(
            *[Encoder1DBlock(embed_dim, 
                             mlp_hidden_dim, 
                             num_heads, 
                             dropout_rate, 
                             attention_dropout_rate) 
              for _ in range(num_layers)])

    def forward(self, x:Tensor):

        assert x.ndim == 3  # (batch, len, emb)

        # Add Position Embedding
        x = self.add_position_embedding(x)
        x = self.dropout(x)

        # Input Encoder
        x = self.encoder_layers(x)
        x = self.norm(x)

        return x