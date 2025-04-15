import torch
from torch.nn import functional as F
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):

        return self.embedding(x) * self.scale

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        pe = torch.zeros(seq_len, d_model)
    
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_t   = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
    
        pe[:,0::2] = torch.sin(position * div_t)
        pe[:,1::2] = torch.cos(position * div_t)
    
        pe = pe.unsqueeze(0)
    
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x+(self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1)) # Learnable Parameter 
        self.beta  = nn.Parameter(torch.zeros(1)) # Learnable Parameter

    def forward(self, x):
        mean = x.mean(dim = -1 , keepdim=True)
        std  = x.std(dim = -1, keepdim=True)

        return self.gamma * (x - mean)/(std + self.eps) + self.beta

class FeedForwardLayer(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Weight and Bias of first Layer W1 and B1
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # weight and Bias of Second Layer W2 and B2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int , dropout: float):
        super().__init__()
        self.d_model = d_model
        self.head = h
        assert d_model % h == 0 , "Dimention of model not divisible by no of head size"

        self.d_k = d_model // h
        self.d_v = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def SelfAttention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]
        attn = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)

        return (attn @ value) , attn

    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1],self.head, self.d_k).transpose(1,2) # size is (batch,h,seq_len,d_k)
        key = key.view(key.shape[0],key.shape[1],self.head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.head, self.d_k).transpose(1,2)

        x, self.attn = MultiHeadAttention.SelfAttention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head*self.d_v )

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attn_blk: MultiHeadAttention, ffb : FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attn_blk = self_attn_blk
        self.ffb = ffb
        self.residual_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn_blk(x, x, x, src_mask))
        x = self.residual_conn[1](x,self.ffb)
        return x 

class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attn_blk: MultiHeadAttention, cross_attn_blk: MultiHeadAttention, ffb : FeedForwardLayer,dropout: float) -> None:
        super().__init__()
        self.self_attn_blk = self_attn_blk
        self.cross_attn_blk = cross_attn_blk
        self.ffb = ffb
        self.residual_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn_blk(x, x, x, tgt_mask))
        x = self.residual_conn[1](x, lambda x: self.cross_attn_blk(x, encoder_op, encoder_op, src_mask))
        x = self.residual_conn[2](x, self.ffb)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_op, src_mask, tgt_mask)
        return self.norm(x)

class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.Linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.Linear(x), dim = -1)
    
class Transformer_Block(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, En_emb: InputEmbeddings, De_emb: InputEmbeddings, En_pos: PositionalEncoding, De_pos: PositionalEncoding, Linear: LinearLayer) ->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.En_emb  = En_emb
        self.De_emb  = De_emb
        self.En_pos  = En_pos
        self.De_pos  = De_pos
        self.Linear  = Linear

    def encode(self, inp, src_mask):
        inp = self.En_emb(inp)
        inp = self.En_pos(inp)
        return self.encoder(inp, src_mask)
    
    def decode(self, encoder_op, src_mask, tgt, tgt_mask):
        tgt = self.De_emb(tgt)
        tgt = self.De_pos(tgt)
        return self.decoder(tgt, encoder_op, src_mask, tgt_mask)
    
    def linear(self,x):
        return self.Linear(x)

def make_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, N: int = 6, d_model: int = 512, h: int = 8 ,dropout: float = 0.01, d_ff: int = 2048) ->None:

    src_emb = InputEmbeddings(d_model, src_vocab_size)
    tgt_emb = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len,dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_blk = MultiHeadAttention(d_model, h, dropout)
        encoder_ffn = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_blk = EncoderBlock(encoder_self_attn_blk, encoder_ffn, dropout)
        encoder_blocks.append(encoder_blk)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attn = MultiHeadAttention(d_model, h, dropout)
        decoder_ffn = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_blk = DecoderBlock(decoder_self_attn, decoder_cross_attn, decoder_ffn, dropout)
        decoder_blocks.append(decoder_blk)

    encoder = Encoder( nn.ModuleList(encoder_blocks))
    decoder = Decoder( nn.ModuleList(decoder_blocks))

    Linear_proj = LinearLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_emb, tgt_emb, src_pos, tgt_pos, Linear_proj)


    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer


