import torch
from torch import nn, einsum
import torch.nn.functional as F
from transformers import AutoModel

from einops import rearrange
from einops.layers.torch import Rearrange

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GatedCrossAttentionBlock, self).__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)  # Additional projection layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.tanh1 = nn.Tanh()
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.tanh2 = nn.Tanh()

    def forward(self, x, feature):
        # Project the feature to match the embedding dimension
        feature_projected = self.projection(feature)
        
        # Cross Attention: Queries from previous conformer block, Keys and Values from the feature
        attn_output, _ = self.cross_attention(query=x, key=feature_projected, value=feature_projected)
        
        # Apply Tanh activation
        x = self.tanh1(attn_output)
        
        # Feed Forward Network
        x = self.ffn(x)
        
        # Apply Gating with Tanh
        x = self.tanh2(x)
        
        return x

class CustomModel(nn.Module):
    def __init__(self, conformer_block, num_features, embed_dim, num_heads, num_labels,bert_dir):
        super(CustomModel, self).__init__()
        self.num_features = num_features
        self.conformer_blocks = nn.ModuleList([conformer_block(embed_dim) for _ in range(num_features)])
        self.gated_cross_attention_blocks = nn.ModuleList(
            [GatedCrossAttentionBlock(embed_dim, num_heads) for _ in range(num_features - 1)]
        )
        # donno we need to see this self.bert = AutoModel.from_pretrained(bert_dir)
        self.classification_head = nn.Linear(embed_dim, num_labels)  # 2 neurons for classification
        self.regression_head = nn.Linear(embed_dim, 1)  # 1 neuron for regression

    def forward(self, features):
        # Start with the first feature being the input to the first conformer block
        x = features[0]  # features[0] has shape [batch_size, num_time_steps, embed_dim]

        for i in range(self.num_features - 1):
            # Apply conformer block
            x = self.conformer_blocks[i](x)
            
            # Apply Gated Cross Attention block using the next feature
            x = self.gated_cross_attention_blocks[i](x.transpose(0, 1), features[i+1].transpose(0, 1))
            x = x.transpose(0, 1)  # Transpose back after attention

        # Final conformer block
        x = self.conformer_blocks[-1](x)

        # donno we need to see this x = self.bert(x) 

        # Classification head
        logits = self.classification_head(x)

        # Regression head with Leaky ReLU
        regression_output = F.leaky_relu(self.regression_head(x))

        return logits, regression_output

# Example usage
num_features = 5  # Adjust according to your input features
embed_dim = 128  # Embedding dimension, adjust as per your requirement
num_heads = 4  # Number of attention heads, adjust as per your requirement
num_labels = 2  # For binary classification

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model
# we can use the above link to add a bert there, we need weights main this is this

# or else we need to train a custom transformer encoder and then use it here

# also adding next token prediction task here is very tough