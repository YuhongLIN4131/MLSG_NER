import torch
from .modeing_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from itertools import chain
import numpy as np
import os
import  copy
import random
MAX_Interval={"Onto":1,"conll03":1,"ace2004":1,"ace2005":1,'genia':1,'cadec':30,'share2013':30,'share2014':30}
MLA_dim={"Onto":64,"conll03":64,"ace2004":64,"ace2005":64,'genia':64,'cadec':96,'share2013':96,'share2014':96}
DISCONTINUE_ENTITY={"cadec":1,"share2013":1,"share2014":1}
def tokenizess(tokenizer, word):
    bpes1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True))
    return bpes1

class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_size,output_size):
        super(MLPWithLayerNorm, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.non_lin1 = F.gelu
        self.layer_norm1 = BertLayerNorm(input_size, eps=1e-12)
        self.linear2 = nn.Linear(input_size, output_size)


    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class CLN(nn.Module):
    def __init__(self, hidden_size):  ###alpha默认0.2
        super(CLN, self).__init__()
        self.hidden_size =hidden_size
        self.rlayer = nn.Linear(self.hidden_size, self.hidden_size)
        self.ulayer = nn.Linear(self.hidden_size, self.hidden_size)
        self.variance_epsilon = 1e12

    def forward(self, context, query):
        r = self.rlayer(context)
        mm = self.ulayer(context)
        u = query.mean(-1, keepdim=True)
        s = (query - u).pow(2).mean(-1, keepdim=True)
        x = r*((query - u) / torch.sqrt(s + self.variance_epsilon)) +mm
        return x

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1,n_in2=None,bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_in2 = n_in2 if n_in2 is not None else n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y

        weight = torch.zeros(( self.n_in2 + int(bias_x),self.n_out, self.n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if len(x.size())==3:#NNW
            if self.bias_x:
                x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
            if self.bias_y:
                y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
            if self.n_out==1:
                s = torch.einsum('bxi,ioj,byj->bxyo', x, self.weight, y).squeeze(-1)
            else:
                # 常规方式
                batch_size,x_size,vector_set_1_size  = x.size()
                y_size= y.size(1)
                # [b, seq_len, v1] -> [b * seq_len, v1]
                vector_set_1 = x.reshape((-1, vector_set_1_size))
                # [v1, class_size, v2] -> [v1, class_size * v2]
                bilinear_map = self.weight.reshape((vector_set_1_size, -1))
                # [b * seq_len, v1] x [v1, class_size * v2] -> [b * seq_len, class_size * v2]
                bilinear_mapping = torch.matmul(vector_set_1, bilinear_map)
                # [b * seq_len, class_size * v2] -> [b, seq_len * class_size, v2]
                bilinear_mapping = bilinear_mapping.reshape(
                    (batch_size, x_size * self.n_out, vector_set_1_size))
                # [b, seq_len * class_size, v2] x [b, seq_len, v2]T -> [b, seq_len*class_size, seq_len]
                bilinear_mapping = torch.matmul(bilinear_mapping, y.transpose(1, -1))
                # [b, seq_len*class_size, seq_len] -> [b, seq_len, class_size, seq_len]
                bilinear_mapping = bilinear_mapping.reshape(
                    (batch_size, x_size, self.n_out, y_size))
                # bilinear_mapping = torch.einsum('bxi,ioj,byj->bxyo', x1, self.bilinearMap, x2)
                return bilinear_mapping.transpose(-2, -1)
        else:#label
            if self.bias_x:
                x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
            if self.bias_y:
                y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
            s = torch.einsum('bxyi,ioj,bxyj->bxyo', x, self.weight, y).squeeze(-1)
        return s

class Attention(nn.Module):
    def __init__(self, hidden_size,label_size):  ###alpha默认0.2
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = 8
        self.temper = (self.hidden_size) ** 0.5
        self.Q_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.K_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.V_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.biaffine = Biaffine(hidden_size, label_size)
        self.dropout=nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, context,mask_query,word_scores_index=None):
        '''context:[batch*seq,hid]   query:[class_number*query_seq,hid]
        mask_query 存的是哪些不能被attend
        '''
        context_Q = self.Q_layer(context)
        query_K = self.K_layer(context)
        query_V = self.V_layer(context)
        '''使用cat来计算一个注意力'''
        attention = self.biaffine(context_Q,query_K).squeeze(-1)#batch,seq,n
        if word_scores_index is not None:
            attention = attention.gather(index=word_scores_index.unsqueeze(2).expand(-1,-1,context.size(1),-1),dim=3).squeeze(-1)
        # attention = torch.matmul(context_Q, query_K.transpose(1, 2))/self.temper
        #用全连接层算
        #一个点，是否只让标签编码attend当前可能的编码，mask_query.unsqueeze(1)也即让每个享有一样的mask
        if len(mask_query.size())==2:
            attention=attention.masked_fill(~mask_query.unsqueeze(1),-10000)#填充屏蔽不可能的标签
        else:#3维度
            attention = attention.masked_fill(~mask_query, -10000)  # 填充屏蔽不可能的标签
        attention = nn.Softmax(dim=-1)(attention)
        attention = self.dropout(attention)
        '''外部注意力的标准正则化方式'''
        out = torch.matmul(attention, query_V)  # batch_size*seq_len,
        #残差连接
        # out = context + self.dropout(out)
        #正则化
        # out = self.norm(out)
        return out
class Attention_layer(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, mask):
        '''embed为查询query的编码，src为context经过embedding后的编码
        ，mask为src的填充字符掩码'''
        # embed[num_query,1024*2]
        for lid, layer in enumerate(self.layers):
            tgt = layer(tgt, mask)

        return tgt

def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    # assert attention_mask.dim() == 2
    return attention_mask.eq(0)

class SSNTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=1024, d_ffn=1024, dropout=0.1, activation="relu", n_heads=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = n_heads
        # self attention
        self.head_dim = d_model//n_heads
        self.scaling = self.head_dim ** 0.5
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        # self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.biaffine = Biaffine(self.head_dim, 1)

        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos



    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, tgt, mask):
        # self attention
        tgt_len, bsz, embed_dim = tgt.size()
        q = self._shape(self.q_proj(tgt),tgt_len, bsz)
        k = self._shape(self.k_proj(tgt),tgt_len, bsz)
        v = self._shape(self.v_proj(tgt),tgt_len, bsz)
        src_len = k.size(1)
        #开始多头的注意力
        # attn_weights = torch.bmm(q, k.transpose(1, 2))/self.scaling
        attn_weights = self.biaffine(q,k).squeeze(-1)
        # attn_weights = self.biaffine(q, k).squeeze(-1)  # batch,seq,n
        #加上mask
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if mask.dim() == 2:
            reshaped = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            reshaped = mask.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        #得到权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #残差连接
        # tgt = tgt + self.dropout2(self.out_proj(attn_output))
        # tgt = self.norm2(tgt)
        # # ffn
        tgt = self.forward_ffn(attn_output)#前馈层结果

        return attn_output
class SSNTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        dropout=0.1
        self.d_model=d_model
        # self.norm2 = nn.LayerNorm(d_model)
        # self.W1 = nn.Parameter(torch.Tensor(d_model, d_model * 2))
        # self.U1 = nn.Parameter(torch.Tensor(d_model, d_model * 2))
        # self.bias1 = nn.Parameter(torch.Tensor(d_model * 2))
        # ffn
    #     self.linear1 = nn.Linear(d_model, d_model)
    #     self.activation = F.relu
    #     self.dropout3 = nn.Dropout(dropout)
    #     self.linear2 = nn.Linear(d_model, d_model)
    #     self.dropout4 = nn.Dropout(dropout)
    #     self.norm3 = nn.LayerNorm(d_model)
    #     self.dropout2 = nn.Dropout(dropout)
    # def forward_ffn(self, tgt):
    #     tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    #     tgt = tgt + self.dropout4(tgt2)
    #     tgt = self.norm3(tgt)
    #     return tgt
    def forward(self, tgt, mask):
        '''embed为查询query的编码，src为context经过embedding后的编码
        ，mask为src的填充字符掩码'''
        # embed[num_query,1024*2]
        tgt = tgt.transpose(0, 1)
        mask = invert_mask(mask)
        for lid, layer in enumerate(self.layers):
            # tgt = self.forward_ffn(layer(tgt, mask))
            tgt = layer(tgt, mask)
            #用lstm的遗忘网络来代替
            # gates1 = tgt @ self.W1 + attn_output @ self.U1 + self.bias1
            # i_t1, f_t1 = (
            #     torch.sigmoid(gates1[:, :, :self.d_model]),  # input
            #     torch.sigmoid(gates1[:, :, self.d_model:]),  # forget
            # )
            # tgt = f_t1 * tgt + i_t1 * attn_output
            # # tgt = tgt + self.dropout2(self.out_proj(attn_output))
            # tgt = self.norm2(tgt)
            # # # ffn
            # tgt = self.forward_ffn(tgt)#前馈层结果
        return tgt.transpose(0, 1)

class wieght_layer(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(wieght_layer, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id,use_decoder=True,use_cat=True,OOV_Integrate=True,label_ids=None
                ,target_type='word',dataset_name=None,label_Attend=True,use_biaffine1=True,use_distance_embedding=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)  # 返回矩阵上三角部分，其余部分定义为0
        # 如果diagonal为空，输入矩阵保留主对角线与主对角线以上的元素；
        # 如果diagonal为正数n，输入矩阵保留主对角线与主对角线以上除去n行的元素；（上三角不要对角线）
        # 如果diagonal为负数 - n，输入矩阵保留主对角线与主对角线以上与主对角线下方h行对角线的元素；
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.use_decoder = use_decoder
        self.use_cat = use_cat
        self.dataset_name = dataset_name
        self.target_type = target_type
        self.three_decoder = False
        self.label_Attend=label_Attend
        self.use_biaffine1=use_biaffine1
        self.use_distance_embedding=use_distance_embedding
        self.OOV_Integrate = OOV_Integrate
        '''让数据集不为非连续时，不需要距离编码的'''
        self.label_ids = label_ids
        mapping = torch.LongTensor([0,1]+label_ids)  #存储特殊符号的token,
        self.register_buffer('mapping', mapping)
        '''可以增加一个关于OOV分词的编码，若是分词则只能'''
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.hid_size=128
        if use_biaffine1:
            self.label_biaffine = Biaffine(hidden_size, 1)
            # self.label2_biaffine = Biaffine(hidden_size, 1)
            # self.label_biaffine = Biaffine(hidden_size, len(label_ids))
        if self.use_cat:
            # self.label_biaffine2 = Biaffine(hidden_size, self.hid_size)
            # self.CLN = LayerNorm(hidden_size, hidden_size, conditional=True)
            # self.CLN = CLN(hidden_size)
            # self.yasuo = MultiNonLinearClassifier(hidden_size,self.hid_size,0.1)
            # decoder_layer = SSNTransformerDecoderLayer(hidden_size, d_ffn=hidden_size, dropout=0.1,n_heads=8)
            decoder_layer = Attention(hidden_size, 1)
            # self.Attend = SSNTransformerDecoder(decoder_layer=decoder_layer, num_layers=3,d_model=hidden_size)
            # self.Attend = SSNTransformerDecoder(self.hid_size,len(label_ids))
            #
            self.Attend = Attention_layer(decoder_layer,3)
            '''使用LSTM吸收标签间的交互信息'''
            # self.birnn = nn.LSTM(hidden_size,self.hid_size, num_layers=1, bidirectional=True,
            #                      batch_first=True)
            self.distance_embedding = nn.Embedding(2, 1024)
            if use_distance_embedding:
                 # 限制长度为20，
                self.special_output = MultiNonLinearClassifier(self.hid_size+10, 1, 0.4)
            else:
                self.special_output = MultiNonLinearClassifier(hidden_size, 1, 0.4)



    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]  ##得到特殊符号的token映射

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:  #
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                ##一个上三角矩阵，每个字不去attend之后的字
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        # hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)

        # 首先计算的是特殊符号
        '''对于特殊符号的话，输出是固定的，可以用网络判断'''
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[
                                            self.label_start_id:self.label_end_id])  # bsz x max_len x num_class
        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state,test=False):
        logits = self(tokens, state,test)


        return logits[:, -1]


class CaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, use_decoder=True,use_cat=True,OOV_Integrate=True,label_ids=None,target_type='word',
                 dataset_name=None,label_Attend=True,use_biaffine1=True,use_distance_embedding=True
                 ):
        super().__init__(decoder, pad_token_id,use_decoder=use_decoder,use_cat=use_cat,OOV_Integrate=OOV_Integrate,label_ids=label_ids,
                         target_type=target_type,dataset_name=dataset_name,label_Attend=label_Attend,use_biaffine1=use_biaffine1,use_distance_embedding=use_distance_embedding)
        self.dropout_layer = nn.Dropout(0.33)

    def forward(self, tokens, state,test=False, Discriminator=False):
        bsz, max_len = tokens.size()
        src_encoder_outputs = state.org_embedding  # batch_size,seq,1024 这个是encoder的输出编码
        bsz, seq_len, hidden_size = src_encoder_outputs.size()
        src_encoder_pad_mask = state.org_mask #这个是encoder的输出编码对应的mask
        src_seq_len = state.encoder_mask.long().sum(dim=-1, keepdim=True)
        '''使用分词头token编码'''
        tgt_seq_len = state.tgt_seq_len
        mask_query = state.mask_query  # 屏蔽所有的query词
        src_mask1 = mask_query  # 不等于填充的值
        if self.target_type == "word":
            all_embedding = torch.cat((
            #                            self.decoder.embed_tokens.weight[0:1].unsqueeze(0).expand(len(state.encoder_output),-1,-1),
            #                            self.decoder.embed_tokens.weight[2:3].unsqueeze(0).expand(len(state.encoder_output),-1,-1),
                                       state.encoder_output[:, 0:1, :],
                                       state.encoder_output.gather(
                                           index=(src_seq_len - 1).unsqueeze(2).expand(-1, -1, hidden_size), dim=1),
                                       self.decoder.embed_tokens.weight[self.mapping[2:]].unsqueeze(0).expand(bsz, -1,
                                                                                                              -1),
                                       state.encoder_output), dim=1)
        else:
            all_embedding = torch.cat((state.encoder_output[:, 0:1, :],
                                       state.encoder_output.gather(
                                           index=(src_seq_len - 1).unsqueeze(2).expand(-1, -1, hidden_size), dim=1),
                                       self.decoder.embed_tokens.weight[self.mapping[2:]].unsqueeze(0).expand(bsz, -1,
                                                                                                              -1),
                                       state.encoder_output), dim=1)
        target_tokens_output = all_embedding.gather(
            index=tokens.unsqueeze(2).expand(-1, -1, hidden_size),
            dim=1)
        '''为每个条件句子添加实体类别标签'''
        # label_start = self.decoder.embed_tokens.weight[tokens[:, 1:2] - 2]
        # mas  = torch.ones_like(tokens[:, 1:2])==1
        # src_encoder_outputs = torch.cat((label_start,src_encoder_outputs),dim=1)
        # src_encoder_pad_mask =torch.cat((mas,src_encoder_pad_mask),dim=1)
        '''综合encoder的编码和word_embedding'''
        if self.training and test==False:
            '''注意Discriminator时最大长度是不包括结束符的'''
            decoder_pad_mask = (~seq_len_to_mask(tgt_seq_len, max_len=tokens.size(1)))
            target_tokens_embedding = target_tokens_output
            if Discriminator == False:#有终点填充符号
                decoder_pad_mask = decoder_pad_mask[:,:-1]
                tokens = tokens[:, :-1]
                target_tokens_embedding = target_tokens_embedding[:, :-1]
            now_causal_masks = self.causal_masks[:tokens.size(1), :tokens.size(1)]
            '''使用Taboo随机mask0.1的没有实体的样本，减少漏标损害'''
            dict = self.decoder(input_ids=tokens,
                                target_tokens_embedding=target_tokens_embedding,
                                encoder_hidden_states=src_encoder_outputs,
                                encoder_padding_mask=src_encoder_pad_mask,#掩码模型，屏蔽一部分的非实体位置
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=now_causal_masks,
                                return_dict=True)
            if Discriminator:#只需要预测最后一个即可
                hidden_state = dict.last_hidden_state.gather(index=tgt_seq_len.unsqueeze(1).unsqueeze(2).expand(-1, -1, hidden_size)-1,
                                        dim = 1)
                decoder_mask1 = (~decoder_pad_mask).gather(index=tgt_seq_len.unsqueeze(1)-1,
                                        dim = 1)  # 表示那些decoder是非填充的
                # target_tokens_output = target_tokens_output.gather(index=tgt_seq_len.unsqueeze(1).unsqueeze(2).expand(-1, -1, hidden_size)-1,
                #                         dim = 1)
            else:
                hidden_state = dict.last_hidden_state[:,1:] # 不需要预测第一个字符  bsz x max_len x hidden_size
                decoder_mask1 = (~decoder_pad_mask)[:,1:]  # 表示那些decoder是非填充的
                # target_tokens_output = target_tokens_output[:,1:-1]
        else:
            if state.past_key_values is None:  # 起点信息需要加载
                target_tokens_embedding = target_tokens_output
                past_key_values = None
                dict = self.decoder(input_ids=tokens[:, 0:max_len-1],
                                    target_tokens_embedding=target_tokens_embedding[:, 0:max_len-1],
                                    encoder_hidden_states=src_encoder_outputs,
                                    encoder_padding_mask=src_encoder_pad_mask,  # 掩码模型
                                    decoder_padding_mask=None,
                                    decoder_causal_mask=None,
                                    past_key_values=past_key_values,
                                    use_prompt_cache=True,
                                    return_dict=True)
                past_key_values = dict.past_key_values
            else:
                target_tokens_embedding = target_tokens_output
                past_key_values = state.past_key_values
            # target_tokens_output = target_tokens_output[:, -1:]
            '''等下，映射好像不对'''
            dict = self.decoder(input_ids=tokens,  # 这里的token就是前面预测出来的加上后面填充为0的
                                target_tokens_embedding=target_tokens_embedding,
                                encoder_hidden_states=src_encoder_outputs,
                                encoder_padding_mask=src_encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
            decoder_mask1 = torch.zeros([hidden_state.size(0), hidden_state.size(1)]).to(
                hidden_state.device) == 0  # 表示那些decoder是非填充的
            state.past_key_values = dict.past_key_values

        batch_size, target_len, hidden_len = hidden_state.size()
        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), all_embedding.size(1)),
            fill_value=-1e30)
        batch_size, scr_seq_len, hidden_size = all_embedding.size()
        word_scores_index = (tokens[:, 1:2] - 2).unsqueeze(1).unsqueeze(2).expand(-1, target_len, scr_seq_len, -1)
        if(self.use_biaffine1):
            #候选集对于标签的分数
            word_scores = self.label_biaffine(self.dropout_layer(hidden_state), self.dropout_layer(all_embedding))
            # word_scores += self.label2_biaffine(self.dropout_layer(target_tokens_output), self.dropout_layer(all_embedding))
            #每一种标签使用一种biaffine
            # word_scores = word_scores.gather(index=word_scores_index,dim=3).squeeze(-1)
            #
        else:
            word_scores=torch.einsum('blh,bnh->bln', self.dropout_layer(hidden_state), self.dropout_layer(all_embedding))
        logits[:, :, 1:] = word_scores[:, :, 1:]
        mask_query = mask_query.unsqueeze(1).expand(-1,logits.size(1),-1) == 1  # batch_size,1,seq_len,1024
        '''统一方案'''
        src_mask2 = src_mask1[:, len(self.mapping):]
        embedding_mask = (src_mask2.unsqueeze(1).expand(-1, target_len, -1) == 1) #句子中的填充
        #开始
        if self.training:
            #标签的下一个是结束符或者任意单词，实体单词的下一个一定时结束符或者其后的位置
            if Discriminator:#只要最后一个
                distance = torch.arange(len(self.mapping), scr_seq_len).unsqueeze(0).unsqueeze(1).to(
                    tokens.device) - tokens.gather(index = tgt_seq_len.unsqueeze(1)-1,dim=1).unsqueeze(2)
            else:
                distance = torch.arange(len(self.mapping), scr_seq_len).unsqueeze(0).unsqueeze(1).to(
                    tokens.device) - tokens[:, 1:].unsqueeze(2)
            distance_flag = (distance > 0)
            embedding_mask[:, 1:, :] = embedding_mask[:, 1:, :] & distance_flag[:, 1:, :]  # 第一个不受距离影响
        else:
            # 标签的下一个是结束符或者任意单词，实体单词的下一个一定时结束符或者其后的位置
            distance = torch.arange(len(self.mapping), scr_seq_len).unsqueeze(0).unsqueeze(1).to(
                tokens.device) - tokens[:,-1:].unsqueeze(2) # 连个token之间距离最大为20                                                                                        -1:].unsqueeze(2) # 连个token之间距离最大为20
            distance_flag = (distance > 0)
            if tokens.size(1)>2:#正式生成实体
                embedding_mask = embedding_mask & distance_flag  # 第一个不受距离影响
        mask_query = mask_query & decoder_mask1.unsqueeze(
            2).expand(-1, -1, scr_seq_len)
        mask_query[:, :, len(self.mapping):] = mask_query[:, :,
                                               len(self.mapping):] & embedding_mask  # 只让需要计算的标签attend

        if self.use_cat:
            '''1、得到两部分的mask编码和距离编码，再整合一起'''
            '''仅仅只是可能的标签之间进行attend'''
            '''
                1、换一个方案保留原编码
            '''
            # CLN_Decoder_Encoder = all_embedding.unsqueeze(1).expand(-1,target_len,-1, -1)
            # #使用lstm吸取信息
            # # CLN_Decoder_Encoder = self.label_biaffine2(hidden_state, all_embedding)  # batch_size,
            # hidden_size = CLN_Decoder_Encoder.size(-1)
            # '''应该整合一下，给出位置不为空的点
            #     CLN_Decoder_Encoder [batch,target_len,mask_query_len,1024]
            # '''
            # mask_query = mask_query.view(-1,scr_seq_len)
            # CLN_Decoder_Encoder = CLN_Decoder_Encoder.reshape(-1,hidden_size)
            # CLN_pos = torch.nonzero(mask_query.reshape(-1)).squeeze(-1)
            # Effective_embedding = CLN_Decoder_Encoder[CLN_pos]
            # '''需要把CLN_Decoder_Encoder中对应不为mask的放在前面'''
            # mask_encoder_outputs = seq_len_to_mask(mask_query.long().sum(dim=-1), max_len=scr_seq_len+1)
            # encoder_outputs = torch.zeros([batch_size*target_len*(scr_seq_len+1),hidden_size]).to(mask_query.device)
            # # 长度+2
            # encoder_pos = torch.nonzero(mask_encoder_outputs.view(-1)).squeeze(-1)
            # new_encoder_pos = torch.nonzero(seq_len_to_mask(mask_query.long().sum(dim=-1), max_len=scr_seq_len).view(-1)).squeeze(-1)
            # encoder_outputs[encoder_pos]=Effective_embedding
            # '''然后放最后一个'''
            # target_pos = mask_query.long().sum(dim=-1)+1 + torch.arange(0,batch_size*target_len*(scr_seq_len+1),(scr_seq_len+1)).to(mask_query.device)
            # '''给出每个序列对应的目标embedding[batch_size*target_len*(scr_seq_len+1),hidden_size]'''
            # target_embedding = hidden_state.repeat_interleave(scr_seq_len+1,dim=1).reshape(-1,hidden_size)[target_pos]
            # encoder_outputs[target_pos] = target_embedding
            # encoder_outputs =  encoder_outputs.view(-1,scr_seq_len+1,hidden_size)
            # start_em = hidden_state.reshape(-1,hidden_size).unsqueeze(1)
            # encoder_outputs = torch.cat((start_em,encoder_outputs),dim=1)
            # # 头好放，尾巴不好放
            # #安排位置
            # '''放完之后就需要计算了'''
            # length = mask_query.long().sum(dim=-1)+2#每个里面的长度
            # sorted, indices = torch.sort(length, descending=True)
            # indices_pos = torch.nonzero(sorted>2).squeeze(-1)
            # indices=indices[indices_pos]
            # sorted=sorted[indices_pos]
            # '''给结果添加目标编码，让源和目标进行双向lstm:显示结果为：
            # 目标编码+可能结果+目标编码'''
            # embed_input_x_packed = torch.nn.utils.rnn.pack_padded_sequence(encoder_outputs[indices],
            #                                                                sorted.detach().cpu().numpy().tolist(),
            #                                                                batch_first=True)
            # encoder_outputs_packed, _ = self.birnn(embed_input_x_packed)
            # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed, batch_first=True)
            # outputs = outputs[:,1:-1]#填充的肯定不需要了
            # #
            # # outputs = outputs.reshape(len(indices),-1,len(self.label_ids),hidden_size//2).gather(index=word_scores_index, dim=3).squeeze(-1)
            # # word_scores_2 = self.special_output(torch.cat((outputs.float(),(ATTEND_embed_outputs[indices,0:outputs.size(1)])),dim=-1)).squeeze(-1).float()
            # word_scores_2 = self.special_output(outputs.float()).squeeze(-1).float()
            # result_score = torch.zeros_like(logits)
            # result_score = result_score.view(-1,scr_seq_len).float()
            # result_score[indices,0:word_scores_2.size(1)] = word_scores_2
            # logits=logits.view(-1)
            # logits[CLN_pos] += result_score.reshape(-1)[new_encoder_pos]
            # logits = logits.reshape(batch_size,target_len,scr_seq_len)
            # mask_query = mask_query.reshape(batch_size,target_len,scr_seq_len)
            #
            # total_embedding=torch.zeros([batch_size,target_len,scr_seq_len,hidden_size]).to(mask_query.device).view(-1,hidden_size)
            # total_embedding[posit]=CLN_Decoder_Encoder
            # total_embedding=total_embedding.reshape(batch_size,target_len,scr_seq_len,-1)#
            # total_embedding = self.yasuo(total_embedding)
            #使用biaffine来综合信息
            '''用transformer吸收标签间信息'''
            #第一步得到有哪些embedding
            index = torch.arange(batch_size).repeat_interleave(target_len).to(hidden_state.device)
            pos = torch.nonzero(decoder_mask1.reshape(-1)).squeeze(-1)
            index=index[pos]
            total_embedding = torch.cat((hidden_state.reshape(-1,hidden_size)[pos].unsqueeze(1),(all_embedding[index])[:,1:2],(all_embedding[index])[:,len(self.mapping)+1:]),dim=1)
            #给出每个对应的maskquery
            mask = mask_query.reshape(-1,scr_seq_len)[pos]
            mask = torch.cat((torch.zeros([len(mask),1]).to(mask.device)==0,mask[:,1:2],mask[:,len(self.mapping)+1:]),dim=1)
            type = torch.zeros_like(mask).long()
            type[:,0]=1
            total_embedding +=self.distance_embedding(type)
            '''是在不行就一次放一半'''
            ATTEND_embed = self.Attend(total_embedding, mask)  # 放到两位
            word_scores_2 = self.special_output(ATTEND_embed[:,1:]).squeeze(-1)  # 最后结果
            logits = logits.reshape(-1,scr_seq_len)
            logits[pos][:,1:2] += word_scores_2[:,0:1]
            logits[pos][:,len(self.mapping)+1:] += word_scores_2[:,1:]
            logits = logits.reshape(batch_size, target_len, scr_seq_len)
            #
            #找对应的tgt编码

            # total_embedding=self.label_biaffine2(aim_embedding,all_embedding)#batch_size,
            # # '''通过CLN的编码'''
            # # print(total_embedding.size())
            # if self.label_Attend:#是否进行标签交互操作
            #     '''在这里对于填充的字可以直接删除掉，减少消耗'''
            #     # ATTEND_embed = self.Attend(total_embedding.reshape(-1,scr_seq_len,total_embedding.size(3)),
            #     #                            mask_query.view(-1,scr_seq_len),word_scores_index.reshape(-1,scr_seq_len,1))#放到两位
            #     ATTEND_embed = self.Attend(total_embedding.reshape(-1, scr_seq_len, total_embedding.size(3)),
            #                                mask_query.view(-1, scr_seq_len)
            #                               )  # 放到两位
            #     ATTEND_embed = ATTEND_embed.reshape(batch_size,target_len,scr_seq_len,-1)#整合回来
            # else:
            #     ATTEND_embed=total_embedding
            # if self.use_distance_embedding:
            #     entity_lengths = torch.clamp(entity_lengths, 0, 20)
            #     entity_lengths_embedding = self.distance_embedding(entity_lengths).squeeze(-1)  # 字字距离编码
            #     distance_embedding = torch.cat((entity_lengths_embedding.unsqueeze(2).expand(-1, -1, len(self.mapping),
            #                                                                                  -1),self.distance_embedding(distance)), dim=-2)  # 词词距离编码
            #     word_scores_2 = self.special_output(torch.cat((ATTEND_embed,distance_embedding),dim=-1)).squeeze(-1)#最后结果
            # else:
            #     # print("标志")
            #     # print(ATTEND_embed.size())
            #     word_scores_2 = self.special_output(ATTEND_embed).squeeze(-1)  # 最后结果
            # logits = logits+word_scores_2
            # logits = word_scores_2
        logits = logits.masked_fill(~mask_query,-1e30)  # 必须相连
        #超过的必须为
        # logits[:,:,0] = -1e30
        '''编码需要起点约束'''
        # if self.target_type=="bpe":
        #     logits[:, :, len(self.mapping):] = logits[:, :, len(self.mapping):].masked_fill(must_word, 1e30)  # 必须相连
        #     logits[:, :, 1:len(self.mapping)] =logits[:, :, 1:len(self.mapping)].masked_fill((token_tail_flag.unsqueeze(2)==1), -1e30)
        # logits = logits.masked_fill(~mask_query, -1e30)  ##填充值的分数补上负无穷
        return logits


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, use_decoder=True, use_cat=True, OOV_Integrate=True, label_ids=None,
                    target_type='word',dataset_name=None,label_Attend=True,use_biaffine1=True,use_distance_embedding=True):
        '''需要'''
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)
        '''默认情况下，encoder和decoder共享相同的embedding'''
        encoder = model.encoder  # 并不会编码关于标签的特殊符号
        decoder = model.decoder
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizess(tokenizer, token)
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = []
                text = token[2:-2].split()
                for izs in text:
                    indexes.append(tokenizess(tokenizer, izs))
                indexes = list(chain(*indexes))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                '''改变了decoder 关于标签类别的embedding'''
                model.decoder.embed_tokens.weight.data[index] = embed  ##用这几个词的平均结果初始化这符号词
                model.encoder.embed_tokens.weight.data[index] = embed  ##用这几个词的平均结果初始化这符号词
        encoder = FBartEncoder(encoder)
        decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id,
                                  use_decoder=use_decoder,use_cat=use_cat,OOV_Integrate=OOV_Integrate,label_ids=label_ids,
                                  target_type=target_type,dataset_name=dataset_name,label_Attend=label_Attend,use_biaffine1=use_biaffine1,
                                  use_distance_embedding=use_distance_embedding)
        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, tgt_seq_len=None, mask_query=None
                    ,OOV_con=None,flag=False
                      ):
        encoder_outputs,mask, hidden_states = self.encoder(src_tokens,src_seq_len)
        src_embed = self.decoder.decoder.embed_tokens.weight[src_tokens]
        # src_embed_outputs = hidden_states[0]
        '''输出的结果中用第一层和最后一层的残差'''
        hidden_size = encoder_outputs.size(2)
        #hidden_size[0]就是第一层加上位置编码的结果
        #提出使用任务的适配器模块
        '''
            1-3层的tsanfoermer,用门网络中和信息。
        '''

        '''
        src_embed_outputs:对应的是word_embedding
        org_embedding:记录的是encoder的输出编码
        org_mask:记录的是encoder序列对应的mask;
        decoder_encoder:用于与decoder输出进行点击的encoder编码
        decoder_encoder_mask：与decoder输出进行点击的encoder编码的mask
        '''
        if self.decoder.use_decoder==True and self.decoder.OOV_Integrate==False and self.decoder.target_type=='word':#
            '''使用encoder后的头分词,'''
            org_embedding=encoder_outputs
            org_mask = mask#条件输入的mask
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            OOV_flag = OOV_con >= 0
            decoder_encoder_mask = OOV_flag.sum(dim=-1) > 0
            OOV_con_now = OOV_con_now[:, :, 0]
            decoder_encoder = encoder_outputs.gather(
                index=OOV_con_now.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)#转化给decoder的输入编码
        elif self.decoder.use_decoder==True and self.decoder.OOV_Integrate==True and self.decoder.target_type=='word':#
            ''''使用decoder整合后的结果'''
            '''最大池化操作或者根据 头token计算'''
            OOV_flag = OOV_con >= 0
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            decoder_encoder_mask = OOV_flag.sum(dim=-1) > 0
            batch_size, seq_OOV_lenn, max_oov_len = OOV_con_now.size()
            '''根据attention合并分词'''
            decoder_encoder = encoder_outputs.unsqueeze(1).expand(-1, seq_OOV_lenn, -1, -1).gather(
                index=OOV_con_now.unsqueeze(3).expand(-1, -1, -1, hidden_size), dim=2)
            # Attention = self.decoder.weight_layer(torch.cat((encoder_outputs, encoder_outputs[:,0:1].expand(-1,encoder_outputs.size(1),-1)),
            #                                                 dim=-1)).squeeze(-1)
            # attention_OOV = Attention.unsqueeze(1).expand(-1, seq_OOV_lenn, -1).gather(
            #     index=OOV_con_now, dim=2)  ##现在已经取得了attention
            # attention_OOV = attention_OOV.masked_fill((~OOV_flag), -1e30)
            # attention_OOV = torch.softmax(attention_OOV, dim=-1)
            # decoder_encoder = (decoder_encoder * attention_OOV.unsqueeze(3) * OOV_flag.unsqueeze(3)).sum(dim=-2)
            '''使用最大池化的效果'''
            min_data = torch.min(decoder_encoder) - 1
            decoder_encoder = (OOV_flag.unsqueeze(3) * decoder_encoder + (~OOV_flag.unsqueeze(3)) * min_data).max(dim=2)[0]
            ''''''
            org_embedding = encoder_outputs
            org_mask = mask
        elif self.decoder.target_type=='word':#直接使用原编码
            org_embedding =encoder_outputs#条件
            org_mask =mask
            '''与decoder交互的编码'''
            OOV_con_now = (OOV_con < 0) * 1 + OOV_con
            OOV_flag = OOV_con >= 0
            decoder_encoder_mask = OOV_flag.sum(dim=-1) > 0
            OOV_con_now = OOV_con_now[:, :, 0]
            decoder_encoder = src_embed.gather(
                index=OOV_con_now.unsqueeze(2).expand(-1, -1, hidden_size), dim=1)  ##这个是用来相乘的
        elif self.decoder.target_type=='bpe':#直接使用原编码
            decoder_encoder = encoder_outputs # 条件
            # decoder_encoder = src_embed # 条件
            decoder_encoder_mask = mask
            org_embedding = encoder_outputs
            org_mask = mask
            # '''让OOV的头token变为最大池化'''
        if flag:  # 训练式
            return decoder_encoder, decoder_encoder_mask, mask_query, org_mask, org_embedding
        else:  # 预测式
            state = BartState(decoder_encoder, decoder_encoder_mask,mask_query
                              , org_mask, org_embedding, tgt_seq_len)
            return state


    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, mask_query=None,OOV_con=None,
                all_word_entity_label=None,aim_mask=None,cal_label_mask=None):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        '''为每一个标签都带一个state'''
        decoder_encoder, decoder_encoder_mask, mask_query, org_mask, org_embedding = self.prepare_state(src_tokens, src_seq_len, tgt_seq_len, mask_query, OOV_con,flag=True)
        batch_size, seq_len, target_length = tgt_tokens.size()
        indices = torch.arange(batch_size, dtype=torch.long).to(src_tokens.device)
        indices = indices.repeat_interleave(seq_len)
        all_word_entity_length = tgt_seq_len  # ，batch*ser_len
        all_word_entity_length = all_word_entity_length.view(-1)
        all_train_pos = torch.nonzero(all_word_entity_length > 0).squeeze(-1)#训练标签的所有位置
        all_train_mask_query = mask_query[indices[all_train_pos]]
        decoder_output = decoder_encoder.new_full((len(all_train_pos), target_length-2, mask_query.size(1)),fill_value=-1e30)#所有训练样本的分数
        target_mask = all_train_mask_query
        '''将位置转化为需要训练的位置'''
        all_word_entity_length = all_word_entity_length[all_train_pos]#一维
        indices = indices[all_train_pos]
        tgt_tokens = tgt_tokens.view(-1, target_length)[all_train_pos]
        tgt_seq_len = tgt_seq_len.view(-1)[all_train_pos]
        batch_size, target_number, entity_len, dimss = all_word_entity_label.size()
        all_word_entity_label = all_word_entity_label.view(batch_size * target_number, entity_len, dimss)[
            all_train_pos]
        aim_mask = aim_mask.view(batch_size * target_number, entity_len + 1)[all_train_pos]
        cal_label_mask = cal_label_mask.view(batch_size * target_number, entity_len,-1)[all_train_pos]
        '''化成两块，小于12，大于12'''
        prefix_len = len(self.decoder.mapping)
        if (all_word_entity_length <=10).sum() > 0:#可能也没有
            noo_entity_pos = torch.nonzero(all_word_entity_length<=10).squeeze(-1)
            # if (tgt_seq_len[noo_entity_pos]-2).sum()*mask_query.size(1)>50000:
            #     #分两块处理,一块是mask_qery小的，一块是mask_query大的
            #     print("进入")
            #     mask_lens = mask_query[indices[noo_entity_pos]].sum(dim=-1)
            #     mask_pos = torch.nonzero(mask_lens<=70).squeeze(-1)
            #     noo_entity_pos1 = noo_entity_pos[mask_pos]
            #     pos1 = indices[noo_entity_pos1]
            #     tgt_tokens1 = tgt_tokens[noo_entity_pos1, 0:10]  # 只需要前3个
            #     state = BartState(decoder_encoder[pos1][:,0:70], decoder_encoder_mask[pos1][:,0:70], mask_query[pos1][:,0:70+prefix_len]
            #                       , org_mask[pos1][:,0:70], org_embedding[pos1][:,0:70], tgt_seq_len[noo_entity_pos1])
            #     decoder_output[noo_entity_pos1, 0:8,0:70+prefix_len] = self.decoder(tgt_tokens1, state)  # 问题是有可能有些batch没有机会有起点
            #     #2
            #     mask_pos = torch.nonzero(mask_lens > 70).squeeze(-1)
            #     if len(mask_pos)>0:
            #         noo_entity_pos1 = noo_entity_pos[mask_pos]
            #         pos1 = indices[noo_entity_pos1]
            #         tgt_tokens1 = tgt_tokens[noo_entity_pos1, 0:10]  # 只需要前3个
            #         state = BartState(decoder_encoder[pos1], decoder_encoder_mask[pos1], mask_query[pos1]
            #                           , org_mask[pos1], org_embedding[pos1], tgt_seq_len[noo_entity_pos1])
            #         decoder_output[noo_entity_pos1, 0:8] = self.decoder(tgt_tokens1, state)  # 问题是有可能有些batch没有机会有起点
            # else:
                #分两块处理
            pos1 = indices[noo_entity_pos]
            # mask_lens = mask_query[indices[noo_entity_pos]].sum(dim=-1).max()
            tgt_tokens1= tgt_tokens[noo_entity_pos,0:10]#只需要前3个
            state=BartState(decoder_encoder[pos1], decoder_encoder_mask[pos1], mask_query[pos1]
                            ,org_mask[pos1],org_embedding[pos1],tgt_seq_len[noo_entity_pos])
            decoder_output[noo_entity_pos,0:8] = self.decoder(tgt_tokens1, state)#问题是有可能有些batch没有机会有起点
        # if ((all_word_entity_length >8) & (all_word_entity_length <=18)).sum() > 0:#可能也没有
        #     noo_entity_pos = torch.nonzero((all_word_entity_length >8) & (all_word_entity_length <=18)).squeeze(-1)
        #     pos1 = indices[noo_entity_pos]
        #     tgt_tokens1= tgt_tokens[noo_entity_pos,0:18]#只需要前3个
        #     state=BartState(decoder_encoder[pos1], decoder_encoder_mask[pos1], src_embed_outputs[pos1], mask_query[pos1]
        #                     ,org_mask[pos1],org_embedding[pos1],tgt_seq_len[noo_entity_pos])
        #     decoder_output[noo_entity_pos,0:16] = self.decoder(tgt_tokens1, state)#问题是有可能有些batch没有机会有起点
        '''第二步：3-6长度的'''
        if (all_word_entity_length > 10).sum()>0:
            '''再调用两次一次划分为<=6，另一个是6之后的，用时间换空间（因为有些数据集中的实体数量太多了）'''
            noo_entity_pos2 = torch.nonzero(all_word_entity_length > 10).squeeze(-1)
            pos2 = indices[noo_entity_pos2]
            # mask_lens = mask_query[indices[noo_entity_pos2]].sum(dim=-1).max()
            tgt_tokens2 = tgt_tokens[noo_entity_pos2]  # 只需要前3个

            state = BartState(decoder_encoder[pos2],
                              decoder_encoder_mask[pos2], mask_query[pos2]
                              , org_mask[pos2],
                              org_embedding[pos2], tgt_seq_len[noo_entity_pos2])
            decoder_output[noo_entity_pos2] = self.decoder(tgt_tokens2, state) # 问题是有可能有些batch没有机会有起点
        '''处理预测的结果+对应的标签'''
        mask = aim_mask[:, 2:] == 0
        mask = ~mask
        # tgt_tokens = tgt_tokens[:, 1:].reshape(-1)
        class_num = mask_query.size(1)
        loss_pos = torch.nonzero(mask.view(-1)).squeeze(-1)
        pred = decoder_output.view(-1, class_num)[loss_pos]  # 预测的结果
        total_sentence, entity_len, dims = all_word_entity_label.size()
        '''标签和mask不对应'''
        labels_mask = cal_label_mask[:, 1:]
        labels = all_word_entity_label.view(total_sentence * entity_len, dims)
        cal_label_mask = cal_label_mask.view(total_sentence * entity_len, dims)
        label_loss_pos = torch.nonzero((cal_label_mask > 0).sum(dim=-1) > 0).squeeze(-1)
        labels = labels[label_loss_pos]
        cal_label_mask = cal_label_mask[label_loss_pos]
        assert len(label_loss_pos) == len(loss_pos)
        '''统计生成过程中错误的点，再以错误的点生成一次，这里只计算最终点，错误点的下一个为的所有标签都为0'''
        '''
            这里batch是指所有的实体序列
            所有的标签:tgt_tokens1.unsqueeze(1).expand(-1,size(2),-1)[batch,target_len,target_len]
            预测的结果:decoder_output>0->[batch,target_len,query_len]
            真实的标签:all_word_entity_label [batch,target_len,query_len]
        '''
        # all_seq = tgt_tokens.unsqueeze(1).expand(-1,target_length,-1)#[batch,target_length,target_length]
        # #需要得到每个序列真实对应多少个标签,这里给出每个序列真实的长度
        # each_len = torch.arange(1,target_length+1).unsqueeze(0).expand(len(all_seq),-1).to(src_tokens.device)
        # tgt_pos = torch.arange(0, len(all_seq)).unsqueeze(1).expand(-1, target_length-2).to(
        #     src_tokens.device)
        # '''注意这里舍弃了第一个和最后一个，因为没有预测值'''
        # all_seq = all_seq[:,1:-1]#第一个没有预测值，最后一个也没有
        # each_len = each_len[:,1:-1]#每个序列的长度
        # all_word_entity_label = all_word_entity_label[:,1:,:]#第一个没有标签
        # #随机给出前8的假的标签，让它在里面随机选择
        # judge_scores = torch.clamp(
        #     torch.min(torch.topk(decoder_output, 3, 2, largest=True)[0], dim=-1)[0], -10, 0)
        # fp_all = ((decoder_output>=judge_scores.unsqueeze(2)).long() - all_word_entity_label).eq(1) & labels_mask#原本没有，但是被预测出来有[batch,target_len-2,query_len]
        # ##需要
        # fp_all[:,:,1] = False#不纠正提前结束的问题
        # '''看看每个长度的值'''
        # #得到对应的新值
        # pred_num = torch.arange(0, fp_all.size(2)).unsqueeze(0).unsqueeze(1).expand(fp_all.size(0), fp_all.size(1), -1).to(src_tokens.device)
        # pos = torch.nonzero(fp_all.long().view(-1)).squeeze(-1)#得到哪些位置是预测错误的
        # #print(len(pos))
        # if len(pos)>0:#真滴有预测错误的
        #     # 只能取前500
        #     pos = pos[torch.randperm(len(pos))[0:200]]
        #     # pos = pos[100:500]
        #     #得到需要进一步计算的下一个点
        #     pred_num = pred_num.view(-1)[pos]#需要进一步计算的点
        #     fp_label_mask = torch.arange(0, class_num).unsqueeze(0).expand(len(pred_num),-1).to(src_tokens.device) > pred_num.unsqueeze(1)
        #     #接下来就是整合新序列了，需要先得到旧序列，放入新值
        #     all_seq = all_seq.unsqueeze(2).expand(-1,-1,class_num,-1)
        #     all_seq = all_seq.reshape(-1,target_length)[pos]#这里就是旧序列
        #     #得到需要计算的旧序列之后就开始加入新序列,
        #     #第一步，根据each_len得到flag矩阵，得到满足位置的值
        #     each_len = each_len.unsqueeze(2).expand(-1,-1,class_num)
        #     each_len = each_len.reshape(-1)[pos]
        #     pad_mask = seq_len_to_mask(each_len, max_len=target_length)
        #     new_all_seq = all_seq * pad_mask + (~pad_mask) * pred_num.unsqueeze(1).expand(-1,target_length)
        #     each_len = each_len+1 #真实长度加1，因为填充了一位
        #     pad_mask = seq_len_to_mask(each_len, max_len=target_length)
        #     new_all_seq = new_all_seq * pad_mask + (~pad_mask) * 1 #填充结果
        #     #new_all_seq就是填充了预测错误值的序列，现在用这个缝合的序列去预测
        #     '''有可能这里给错了'''
        #     tgt_pos = tgt_pos.unsqueeze(2).expand(-1,-1,class_num)
        #     tgt_pos = tgt_pos.reshape(-1)[pos]
        #     pos_index = indices[tgt_pos]
        #     # tgt_tokens2 = new_all_seq #
        #     # state = BartState(decoder_encoder[pos2], decoder_encoder_mask[pos2],
        #     #                   src_embed_outputs[pos2], mask_query[pos2]
        #     #                   , org_mask[pos2], org_embedding[pos2],
        #     #                   each_len)
        #     #处理
        #     fp_pred = torch.zeros([len(pos),class_num]).to(src_tokens.device)*-1e30
        #     # '''打包成两批处理'''
        #     if (each_len <= 12).sum() > 0:  # 可能也没有
        #         noo_entity_pos = torch.nonzero(each_len <= 12).squeeze(-1)
        #         pos1 = pos_index[noo_entity_pos]
        #         tgt_tokens1 = new_all_seq[noo_entity_pos, 0:12]  # 只需要前3个
        #         state = BartState(decoder_encoder[pos1], decoder_encoder_mask[pos1], src_embed_outputs[pos1],
        #                           mask_query[pos1]
        #                           , org_mask[pos1], org_embedding[pos1], each_len[noo_entity_pos])
        #         fp_pred[noo_entity_pos] = self.decoder(tgt_tokens1, state,Discriminator=True).squeeze(1)  # 问题是有可能有些batch没有机会有起点
        #     '''第二步：3-6长度的'''
        #     if (each_len > 12).sum() > 0:
        #         '''再调用两次一次划分为<=6，另一个是6之后的，用时间换空间（因为有些数据集中的实体数量太多了）'''
        #         noo_entity_pos2 = torch.nonzero(each_len > 12).squeeze(-1)
        #         pos2 = pos_index[noo_entity_pos2]
        #         tgt_tokens2 = new_all_seq[noo_entity_pos2]  # 只需要前3个
        #         state = BartState(decoder_encoder[pos2], decoder_encoder_mask[pos2],
        #                           src_embed_outputs[pos2], mask_query[pos2]
        #                           , org_mask[pos2], org_embedding[pos2],
        #                           each_len[noo_entity_pos2])
        #         fp_pred[noo_entity_pos2] = self.decoder(tgt_tokens2, state,Discriminator=True).squeeze(1)  # 问题是有可能有些batch没有机会有起点
        #     '''分两批处理结束'''
        #     # fp_pred = self.decoder(tgt_tokens2, state,Discriminator=True).squeeze(1) # 鉴别式
        #     pred = torch.cat((pred,fp_pred),dim=0)
        #     #填充fp的label，统一都是0
        #     fp_labels = torch.LongTensor([0]*labels.size(1)).unsqueeze(0).expand(len(fp_pred),-1).to(src_tokens.device)
        #     labels = torch.cat((labels,fp_labels),dim=0)
        #     # 填充fp的cal_label_mask,只有大于当前的 & mask_query 才需要被学习。
        #     fp_label_mask[:,1] = True
        #     fp_label_mask = mask_query[pos_index].bool() & fp_label_mask
        #     cal_label_mask =torch.cat((cal_label_mask,fp_label_mask),dim=0)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': pred,'target_mask':target_mask,'cal_label_mask':cal_label_mask,'labels':labels}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, mask_query,org_mask,org_embedding,tgt_seq_len):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        # self.src_embed_outputs = src_embed_outputs
        self.mask_query = mask_query
        self.org_mask = org_mask
        self.org_embedding = org_embedding
        self.tgt_seq_len = tgt_seq_len

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        # self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        self.mask_query = self._reorder_state(self.mask_query, indices)
        if self.tgt_seq_len is not None:
            self.tgt_seq_len = self._reorder_state(self.tgt_seq_len, indices)
        # self.NNW_score = self._reorder_state(self.NNW_score, indices)
        # if self.bpe_tail_flag is not None:
        #     self.bpe_tail_flag = self._reorder_state(self.bpe_tail_flag, indices)
        #     self.bpe_head_flag = self._reorder_state(self.bpe_head_flag, indices)
        if self.org_mask is not None:
            self.org_mask = self._reorder_state(self.org_mask, indices)
            self.org_embedding = self._reorder_state(self.org_embedding, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new