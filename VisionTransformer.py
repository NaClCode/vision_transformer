from torch import nn
from torch import Tensor
from EncoderBlock import Encoder
from PatchEmbed import PatchEmbed
from functools import partial
import torch
"""VisionTransformer."""

class VisionTransformer(nn.Module):
    '''
    Vision Transformer is the complete end to end model architecture which combines all the above modules
    in a sequential manner. The sequence of the operations is as follows
    '''
    def __init__(self,
                 img_size = 24,
                 patch_size = 16,
                 in_dim = 3,
                 num_classes = 200,
                 embed_dim = 768,
                 heads = 12,
                 num_layers = 12,
                 mlp_ratio = 4,
                 dropout_rate = 0., 
                 attention_dropout_rate = 0.,
                 add_position_embedding = True,
                 representation_size = None,
                 classifier = 'token',
                 norm_layer = None,
                 act_layer = None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.classifier = classifier


        self.patch_embed = PatchEmbed(img_size = img_size,
                                      patch_size = patch_size,
                                      in_dim = in_dim, 
                                      embed_dim = embed_dim,
                                      norm_layer = norm_layer)
        
        self.patches = self.patch_embed.num_patches

        if self.classifier in ['token']:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.patches = self.patches + 1
        
        
        self.encoder = Encoder(embed_dim = embed_dim, 
                               num_layers = num_layers, 
                               mlp_hidden_dim = mlp_ratio * embed_dim, 
                               num_heads = heads, 
                               dropout_rate = dropout_rate, 
                               attention_dropout_rate = attention_dropout_rate, 
                               add_position_embedding = add_position_embedding,
                               num_patches = self.patches)
        
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
    
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    
    def forward(self, x:Tensor):
        x = self.patch_embed(x)

        if self.classifier in ['token']:
            cls = torch.tile(self.cls, [x.shape[0], 1, 1])
            x = torch.cat((cls, x), dim = 1)

        x = self.encoder(x)

        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = torch.mean(x, dim = list(range(1, x.ndim - 1))) 
        
        x = self.pre_logits(x)
        x = self.head(x)
        return x
