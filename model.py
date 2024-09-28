import torch
from torch import nn, einsum
import torch.nn.functional as F
from transformers import AutoModel

from einops import rearrange
from einops.layers.torch import Rearrange

class CustomModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_labels,bert_dir,input_dims,update_bert):
        super(CustomModel, self).__init__()
        self.num_features = len(input_dims)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(dim=embed_dim) for _ in range(self.num_features)])
        self.gated_cross_attention_blocks = nn.ModuleList(
            [GatedCrossAttentionBlock(embed_dim, num_heads) for _ in range(self.num_features - 1)]
        )
        self.projection_blocks = nn.ModuleList([nn.Conv1d(input_dim, embed_dim, kernel_size=1) for input_dim in input_dims])

        # self.bert = CustomBERT(bert_dir)
        # if not update_bert:
        #     for param in self.bert.parameters():
        #         param.requires_grad = False

        self.bert_projection = nn.Linear(embed_dim,768)
        self.classification_head = nn.Linear(embed_dim, num_labels)  
        self.regression_head = nn.Linear(embed_dim, 1)

        # if asr:
        #     self.decoder_layer = nn.TransformerDecoderLayer(embed_dim,num_heads,batch_first=True)
        #     self.decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=1)
        #     self.lm_head = nn.Linear(embed_dim,num_vocab)

    def forward(self, fbank_features, wav2vec2_features, egmap_features, trill_features, phonetic_features):
        # Start with the first feature being the input to the first conformer block
        x = fbank_features  # [bs,nt,h]
        #print(f'[FBANK]{x.shape}')
        # x = x.permute(0, 2, 1)
        # x = self.projection_blocks[0](x)  # only project when we are not using 512 mels
        # x = x.permute(0, 2, 1)
        for i in range(self.num_features - 1):
            x = self.conformer_blocks[i](x)
            #x = x.permute(0,2,1) # [bs,h,nt]
            #print(f"[1st projection]{x.shape}")
            projected_feature = self.projection_blocks[i]([wav2vec2_features, egmap_features, trill_features, phonetic_features][i].permute(0,2,1))
            #print(f"[projected feature]{projected_feature.shape}")
            projected_feature = projected_feature.permute(0, 2, 1)
            #print(f"[projected feature after permute]{projected_feature.shape}")
            x = self.gated_cross_attention_blocks[i](x, projected_feature)
            #print(f"[After x_attenion]{x.shape}")
            #x = x.permute(0, 2, 1)
            #print(f"[2nd projection]{x.shape}")

        # Final conformer block
        x = self.conformer_blocks[-1](x)
        # Pass through bert for textual understanding
        # x = self.bert_projection(x)
        #speech_embeddings = x
        x = x.mean(dim=1)

        # if asr:
        #     tgt = self.pos_embedding(tgt)
        #     tgt = self.token_embedding(tgt)
        #     txt_feats = self.decoder(speech_embeddings,tgt)
        #     token_logits = self.lm_head(txt_feats)

        # _, x = self.bert(x) 

        # Classification head
        logits = self.classification_head(x)

        # Regression head
        regression_output = F.leaky_relu(self.regression_head(x))
        
        speech_embeddings = self.bert_projection(x)
        return logits, regression_output, speech_embeddings
    
class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GatedCrossAttentionBlock, self).__init__()
    
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first = True)
        self.tanh1 = nn.Tanh()
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.tanh2 = nn.Tanh()

    def forward(self, x, feature):
        # Cross Attention: Queries from previous conformer block, Keys and Values from the feature
        attn_output, _ = self.cross_attention(query=x, key=feature, value=feature)
        x = self.tanh1(attn_output)
        x = self.ffn(x)
        x = self.tanh2(x)
        
        return x
    
# lucidrains/conformer
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

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

class CustomBERT(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, custom_input):
        # Bypass the embedding layer
        # Custom input shape: [batch_size, num_time_steps, feature_dim]
        
        # Pass custom input directly into the encoder
        bert_output = self.bert.encoder(
            custom_input,
            attention_mask=None  # No attention mask needed
        )

        pooled_output = bert_output[0][:, 0]
        return bert_output[0], pooled_output
