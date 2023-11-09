import torch
from torch import nn
from torch import Tensor
"""
Position embeddings are added to the patch embeddings to retain positional information.
"""
"""
class AddPositionEmbs(nn.Module):

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe

    
if self.add_position_embedding:
    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(x)
"""
class AddPositionEmbs(nn.Module):

    def __init__(self, num_patches, embed_dim) -> None:
        super(AddPositionEmbs, self).__init__()
        pos_emb_shape = (1, num_patches, embed_dim)
        self.pe = nn.Parameter(torch.randn(pos_emb_shape))

    def forward(self, inputs:Tensor):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        
        return inputs + self.pe