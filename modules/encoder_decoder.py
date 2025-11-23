from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pad_tokens(att_feats):
    # ---->pad
    H = att_feats.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    att_feats = torch.cat([att_feats, att_feats[:, :add_length, :]], dim=1) 
    return att_feats

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def process_features(src, short_text, m, k):

    n = src.size(1) 
    w = (n + m- 1) // m
    padding = m* w - n
    src_padded = F.pad(src.squeeze(0), (0, 0, 0, padding), value=0)
    src_groups = src_padded.view(w, m, 512)
    src_avg = src_groups.mean(dim=1) 
    cos_sim = F.cosine_similarity(src_avg.unsqueeze(1), short_text.unsqueeze(0), dim=2)  # [w, t]
    topk_sim, topk_idx = cos_sim.topk(k, dim=1, largest=True, sorted=True)  # [w, k]
    topk_features = short_text[topk_idx]  
    topk_avg = topk_features.mean(dim=1)  
    final_feature = topk_avg 
    
    return final_feature


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.down_proj = nn.Linear(512, 512)

      
        
    def forward(self, src, tgt, src_mask, tgt_mask):

        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src_all, src_mask):
        
        x_plip = src_all[:,:,-512:]
        src_uni = src_all[:,:,:-512]

        return self.encoder(self.src_embed(src_uni), src_mask,x_plip)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask)


    
class Encoder(nn.Module):
    def __init__(self, layer, N, PAM,bank_path,v,m,k):
        super(Encoder, self).__init__()
        self.bank_path = bank_path
        self.layers = layer
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N
        self.latents = nn.Parameter(torch.randn(1, 512))
        self.latents_t = nn.Parameter(torch.randn(1, 512))
        self.down_proj = nn.Linear(512, 512)
      
        self.short_text = torch.tensor(torch.load(self.bank_path))
        self.v = v
        self.k = k
        self.m = m
        
    def forward(self, x, mask,x_plip):
       
        
        fu = repeat(self.latents, 'n d -> b n d', b = x.shape[0]) 
        fu_t = repeat(self.latents_t, 'n d -> b n d', b = x.shape[0]) 
        s=[]
        st = []
      
        
        for i in range(self.N):
            if i == 0:
                fu,fu_t, sorted_indices = self.layers(x, mask,fu,fu_t,None)
                s.append(fu)
            elif i == 1:
              
                sorted_plip = x_plip.squeeze(0)[sorted_indices.squeeze(0)] 
               
                x_plip = sorted_plip[:int(sorted_plip.shape[0]*self.v),:].unsqueeze(0)
                m = self.m
                k = self.k
                short_text = self.short_text.to(x.device)
                retrieval_text = process_features(x_plip, short_text, m, k)
                retrieval_text = self.down_proj(retrieval_text)
                retrieval_text = retrieval_text.unsqueeze(0)
               
                fu,fu_t, _  = self.layers(x, mask,fu,fu_t,retrieval_text)
                st.append(fu_t)
                s.append(fu)
            else:
                fu, fu_t, _ = self.layers(x, mask,fu,fu_t,retrieval_text)
                st.append(fu_t)
                s.append(fu)



        
        o = s[0]+s[1]+s[2]
        ot = st[0]+st[1]
        o = torch.cat([o,ot],dim=1)
 
        return o 
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, src_attn, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = clones(feed_forward,2)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 4)
        self.d_model = d_model


    def forward(self, x, mask,fu,fu_t,retrieval_text):
        fu, attention_weights = self.sublayer[0](fu, lambda fu: self.src_attn(fu, x, x, mask=None))
        attention_weights = attention_weights.mean(dim=1)
        attention_values = attention_weights.squeeze(1) 
      
        sorted_indices = torch.argsort(attention_values, dim=-1, descending=True)  
        
        fu = self.sublayer[1](fu, self.feed_forward[0])
        fu,_ = self.sublayer[2](fu, lambda fu: self.self_attn(fu, fu, fu, mask=None))
        fu = self.sublayer[3](fu, self.feed_forward[1])
        if retrieval_text is not None:
            fu_t,_= self.sublayer[0](fu_t, lambda fu_t: self.src_attn(fu_t, retrieval_text, retrieval_text, mask=None))
            fu_t = self.sublayer[1](fu_t, self.feed_forward[0])
            fu_t,_ = self.sublayer[2](fu_t, lambda fu_t: self.self_attn(fu_t, fu_t, fu_t, mask=None))
            fu_t = self.sublayer[3](fu_t, self.feed_forward[1])
      
        return fu,fu_t,sorted_indices

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        norm_x = self.norm(x)
        
     
        output = sublayer(norm_x)
        
       
        if isinstance(output, tuple):
            out, attn = output
            return x + self.dropout(out), attn
        else:
            return x + self.dropout(output)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        m = hidden_states
        x ,_= self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x,_ = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
       
        return self.sublayer[2](x, self.feed_forward)







class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PAM(nn.Module):
    def __init__(self, dim=512):
        super(PAM, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 13, 1, 13//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        B, H, C = x.shape
        assert int(math.sqrt(H))**2==H, f'{x.shape}'
        cnn_feat = x.transpose(1, 2).view(B, C, int(math.sqrt(H)), int(math.sqrt(H))).contiguous()
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        return x


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        pp = PAM(self.d_model)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers,pp,self.bank_path,self.v,self.m,self.k),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.bank_path = args.bank_path
        self.v = args.v
        self.m = args.m
        self.k = args.k

   

        tgt_vocab = self.vocab_size + 1

        self.embeded = Embeddings(args.d_vf, tgt_vocab)
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.logit_mesh = nn.Linear(args.d_model, args.d_model)
        
    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks, meshes= None):
  
  
        att_feats, seq, _, att_masks, seq_mask, _= self._prepare_feature_forward(att_feats, att_masks, meshes)
        
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_mesh(self, att_feats, att_masks=None, meshes=None):


        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None

        return att_feats, meshes, att_masks, meshes_mask

    def _prepare_feature_forward(self, att_feats, att_masks=None, meshes=None, seq=None):
        
        att_feats_plip = att_feats[:,:,-512:]
        att_feats = att_feats[:,:,:-512]

        
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None
        att_feats = torch.cat((att_feats,att_feats_plip),dim=2)
        return att_feats, seq, meshes, att_masks, seq_mask, meshes_mask

    def _forward(self, fc_feats, att_feats,  report_ids, att_masks=None):

        att_feats_uni = att_feats[:,:,:-512]
        att_feats_plip = att_feats[:,:,-512:]
        att_feats_uni, report_ids, att_masks, report_mask = self._prepare_feature_mesh(att_feats_uni, att_masks, report_ids)

        att_feats = torch.cat((att_feats_uni,att_feats_plip),dim=2)
        out = self.model(att_feats,report_ids, att_masks, report_mask)

        outputs = F.log_softmax(self.logit(out), dim=-1)


        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.long().unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))

        return out[:, -1], [ys.unsqueeze(0)]

    def _encode(self, fc_feats, att_feats, att_masks=None):

        att_feats, _, att_masks, _ = self._prepare_feature_mesh(att_feats, att_masks)
        out = self.model.encode(att_feats,att_masks)

        return out