import torch
from torch import nn, einsum
from einops import rearrange
from math import sqrt


class LayerNorm(nn.Module):
    def __init__(self, dim, axis=1, num_axes=4, eps=1e-5):
        """Layer normalization applied over axis 'axis' with the size 'dim'.
        Total number of axes in the inputs and outputs are given by 'num_axes'
    
        Args:
            dim (int): size of the normalized dimension
            axis (int, optional): Over which axis to normalize. Defaults to 1.
            num_axes (int, optional): Number of axes. Defaults to 4.
            eps (float, optional): epsilon. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.axis = axis
        shape = [1]*num_axes
        shape[axis] = dim
        self.w = nn.Parameter(torch.ones(shape))
        self.b = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        std = torch.var(x, dim=self.axis, unbiased=False,keepdim=True).sqrt()
        mean = torch.mean(x, dim=self.axis, keepdim=True)
        return (x - mean) / (std + self.eps) * self.w + self.b
    

class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads, reduction):
        """Efficient spatial reduction multihead attention.

        Args:
            dim (int): dimensions
            num_heads (int): number of heads
            reduction (int): spatial reduction factor for K and V vectors
        """
        super().__init__()
        self.scale = 1.0/sqrt(dim // num_heads)
        self.num_heads = num_heads

        self.w_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.w_k = nn.Conv2d(dim, dim, reduction, stride=reduction, bias=False)
        self.w_v = nn.Conv2d(dim, dim, reduction, stride=reduction, bias=False)
        self.w_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.size()[-2:]

        q, k, v = (self.w_q(x), self.w_k(x), self.w_v(x))
        head_rearrange = lambda t: rearrange(t, 'b (n d) h w -> (b n) d (h w)',
                                             n=self.num_heads)
        q, k, v = map(head_rearrange, (q, k, v))

        attn = einsum('b d i, b d j -> b i j', q, k) * self.scale
        attn = attn.softmax(dim = -1)

        out = einsum('b i j, b d j -> b d i', attn, v)
        out = rearrange(out, '(b n) d (h w) -> b (n d) h w',
                        n=self.num_heads, h=h)
        return self.w_out(out)
    

class DepthwiseConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, padding=padding,
                      groups=c_in, stride=stride, bias=bias),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)
    

class MixFNN(nn.Module):
    def __init__(self, dim, expansion_factor):
        """Mix feed-forward network

        Args:
            dim (int): number of input and output dimensions
            expansion_factor (int): ratio between hidden dimensions and input
                dimensions
        """
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.ff = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.ff(x)


class MiTStage(nn.Module):
    def __init__(self, c_in, c_out, kernel, num_heads, attn_reduction,
                 ff_expansion, num_layers):
        """MiT encoder building block. Normally, an MiT encoder consists of 4
        such blocks, each generating a feature map of different resolution.

        Args:
            c_in (int): number of input channels
            c_out (int): number of output channels
            kernel (int): kernel size of the patch embedding
            num_heads (int): number of attention heads
            attn_reduction (int): attention head spatial reduction factor
            ff_expansion (int): expansion factor of the MixFNN
            num_layers (int): number of attention layers
        """
        super().__init__()
        if kernel != 3 and kernel != 7:
            raise ValueError(f'In MiTStage, supported kernel sizes are 3 and 7,'
                             f' but {kernel} is provided')
        stride = kernel//2 + 1
        padding = stride - 1

        self.patch_embedding = nn.Conv2d(c_in, c_out, kernel, stride, padding)
        self.attention_layers = nn.ModuleList([])
        for _ in range(num_layers):
            attention = nn.Sequential(LayerNorm(c_out),
                                      SpatialReductionAttention(dim=c_out,
                                                                num_heads=num_heads,
                                                                reduction=attn_reduction))
            ff = nn.Sequential(LayerNorm(c_out),
                               MixFNN(dim=c_out, expansion_factor=ff_expansion))
            self.attention_layers.append(nn.ModuleList([attention, ff]))

    def forward(self, x):  # (N, C, H, W)
        x = self.patch_embedding(x)  # (N, C_out, H//S, W//S)

        for (attention, ff) in self.attention_layers:
            x = attention(x) + x  # (N, C_out, H//S, W//S)
            x = ff(x) + x  # (N, C_out, H//S, W//S)

        return x  # (N, C_out, H//S, W//S)


class MiT(nn.Module):
    def __init__(self, name=None, kernels=None, stage_channels=None,
                 num_heads=None, attn_reductions=None, ff_expansions=None,
                 num_layers=None, in_channels=3):
        """MiT encoder for SegFormer

        Args:
            name (str, optional): Name of the MiT model. Can be either
                b0, b2 or b5. Defaults to None.
            kernels (tuple, optional): Tuple of MiT stages' kernel sizes.
                Must be specified if 'name' argument is not provided.
                Defaults to None.
            stage_channels (tuple, optional): Tuple of MiT stages' channel
                numbers. Must be specified if 'name' argument is not provided.
                Defaults to None.
            num_heads (tuple, optional): Tuple of MiT stages' attention head
                numbers. Must be specified if 'name' argument is not provided.
                Defaults to None.
            attn_reductions (tuple, optional): Tuple of MiT stages' attention
                reduction factors. Must be specified if 'name' argument is not
                provided. Defaults to None.
            ff_expansions (tuple, optional): Tuple of MiT stages' feed-forward
                expansions factors. Must be specified if 'name' argument is
                not provided. Defaults to None.
            num_layers (tuple, optional): Tuple of MiT stages' attention layer
                numbers. Must be specified if 'name' argument is not provided.
                Defaults to None.
            in_channels (int, optional): Number of channels in the input.
                Defaults to 3.
        """
        super().__init__()

        if name is not None:
            kernels = (7, 3, 3, 3)
            num_heads = (1, 2, 5, 8)
            attn_reductions = (8, 4, 2, 1)

        if name=='b0':
            stage_channels = (32, 64, 160, 256)
            ff_expansions = (8, 8, 4, 4)
            num_layers = (2, 2, 2, 2)
        elif name=='b2':
            stage_channels = (64, 128, 320, 512)
            ff_expansions = (8, 8, 4, 4)
            num_layers = (3, 3, 6, 3)
        elif name=='b5':
            stage_channels = (64, 128, 320, 512)
            ff_expansions = (4, 4, 4, 4)
            num_layers = (3, 6, 40, 3)
        elif name is not None:
            raise ValueError(f'Only b0, b2 and b5 MiT models are supported,'
                             f' but {name} was provided')
        
        self.stage_channels = stage_channels
        c_out = stage_channels
        c_in = (in_channels,) + c_out[:-1]

        args = (c_in, c_out, kernels, num_heads, attn_reductions,
                ff_expansions, num_layers)
        
        if name is None:
            if sum([arg is None for arg in args]):
                raise ValueError('For customly configured MiT modules, all '
                                 'arguments except name and in_channels must '
                                 'be provided ')
            if sum([len(arg)!=len(kernels) for arg in args]):
                raise ValueError('All the arguments in MiT except the name and'
                                 ' in_channels must be of the same length')

        mit_stages = [MiTStage(*unpacked_args) for unpacked_args in zip(*args)]
        self.stages = nn.ModuleList(mit_stages)

    def forward(self, x, return_stage_outputs=False):
        stage_outputs = []
        for stage in self.stages:
            x = stage(x) 
            stage_outputs.append(x)
        
        return stage_outputs if return_stage_outputs else x
    

class SegFormer(nn.Module):
    def __init__(self, mit=None, num_classes=4, in_channels=3,
                 decoder_dim=None, kernels=None, stage_channels=None,
                 num_heads=None, attn_reductions=None, ff_expansions=None,
                 num_layers=None):
        """SegFormer - transformer-based semantic segmentation module.
        Consists of an MiT encoder and a segmentation head

        Args:
            mit (str, optional): Name of the MiT model. Can be either b0, b2
                or b5. Defaults to None.
            num_classes (int, optional): Number of segmentation classes.
                Defaults to 4.
            in_channels (int, optional): Number of channels in the input.
                Defaults to 3.
            decoder_dim (int, optional): Number of hidden dimensions in the
                segmentation head. Must be specified if 'mit' argument is not
                provided. Defaults to None.
            kernels (tuple, optional): Tuple of MiT stages' kernel sizes.
                Must be specified if 'mit' argument is not provided.
                Defaults to None.
            stage_channels (tuple, optional): Tuple of MiT stages' channel
                numbers. Must be specified if 'mit' argument is not provided.
                Defaults to None.
            num_heads (tuple, optional): Tuple of MiT stages' attention head
                numbers. Must be specified if 'mit' argument is not provided.
                Defaults to None.
            attn_reductions (tuple, optional): Tuple of MiT stages' attention
                reduction factors. Must be specified if 'mit' argument is not
                provided. Defaults to None.
            ff_expansions (tuple, optional): Tuple of MiT stages' feed-forward
                expansions factors. Must be specified if 'mit' argument is not
                provided. Defaults to None.
            num_layers (tuple, optional): Tuple of MiT stages' attention layer
                numbers. Must be specified if 'mit' argument is not provided.
                Defaults to None.
        """
        super().__init__()
        if mit=='b0':
            decoder_dim = 256
        if mit=='b2' or mit=='b5':
            decoder_dim = 768
        if mit is None and decoder_dim is None:
            raise ValueError('If MiT encoder name is not provided (mit argument),'
                             ' all the other arguments must be specified')

        self.mit = MiT(mit, kernels, stage_channels, num_heads,
                       attn_reductions, ff_expansions, num_layers, in_channels)
        dims = self.mit.stage_channels
        fusing = [nn.Sequential(nn.Conv2d(c, decoder_dim, 1),
                                nn.Upsample(scale_factor = 2**i))\
                    for i, c in enumerate(dims)]
        self.fusing = nn.ModuleList(fusing)
        self.segmentation_head = nn.Sequential(nn.Conv2d(4*decoder_dim, decoder_dim, 1),
                                               nn.Conv2d(decoder_dim, num_classes, 1))

    def forward(self, x):
        stage_outputs = self.mit(x, return_stage_outputs=True)
        stage_outputs = [fusing(out) for out, fusing in zip(stage_outputs, self.fusing)]
        fused_outputs = torch.cat(stage_outputs, dim=1)
        return self.segmentation_head(fused_outputs)
