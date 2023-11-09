from torch import nn
from torch import Tensor
class PatchEmbed(nn.Module):
    """
    The standard Transformer receives as input a 1D sequence of token embeddings. 
    To handle 2D images, we reshape the image into a sequence of flattened 2D patches.
    """
    """
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)
    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    """
    """
    这个地方是jax的写法, 而
        `torch.Tensor` 高维矩阵的表示 N x C x H x W
        `numpy.ndarray`高维矩阵的表示 N x H x W x C
        因此在两者转换的时候需要使用`numpy.transpose( )` 方法 。
    """
    def __init__(self, 
                 img_size = 224, #图片大小
                 patch_size = 16, #块大小
                 in_dim = 3, #输入维度
                 embed_dim = 768, #编码维度 16 x 16 x 3
                 norm_layer = None #层正则化
                ):

        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size) # 224 x 224
        self.patch_size = (patch_size, patch_size) # 16 x 16
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])  # 224/16 x 224/16 = 14 x 14
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 14 x 14
        """
        in_channels: 输入特征矩阵的深度
        out_channels: 卷积核的个数
        kernel_size: 卷积核大小
        stride: 卷积步长
        """
        self.conv = nn.Conv2d(in_channels = in_dim, 
                              out_channels = embed_dim, 
                              kernel_size = self.patch_size, 
                              stride = self.patch_size)
       
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() 
        
    def forward(self, x:Tensor):
        n, h, w, c = x.shape 
        """
        n: 图片数
        h: 图片高
        w: 图片宽
        c: 图片深, RGB
        """
        x = self.conv(x)
        """
        N, C, H, W ==> flatten 
        >>> N, C, H * W
        N, C, H * W ==> transpose 
        >>> N, H * W, C
        """
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x) # 把x展平
        return x