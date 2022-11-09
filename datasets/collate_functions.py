# encoding: utf-8

import torch
from typing import List
import numpy as np
import os
import random



def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    ##这个函数好像对于多任务不通用啊
    batch_size = len(batch)
    output = []
    '''对于target的值，可能因为是否划分每个实体类别而维度不同，因此'''
    if len(batch[0][0].size())==1:#直接序列生成
        max_length = max(x[0].shape[0] for x in batch)
        pad_target = torch.ones([batch_size, max_length], dtype=torch.long)
        for sample_idx in range(batch_size):#填充第一个
            data = batch[sample_idx][0]
            pad_target[sample_idx, : data.shape[0]] = data
        output.append(pad_target)
    else:##使用了decoder_prompt或者每个词都生成
        max_length = max(x[0].shape[1] for x in batch)
        label_len = max(x[0].shape[0] for x in batch)
        pad_target = torch.ones([batch_size, label_len,max_length], dtype=torch.long)
        for sample_idx in range(batch_size):  # 填充第一个
            data = batch[sample_idx][0]
            pad_target[sample_idx, : data.shape[0],: data.shape[1]] = data
        output.append(pad_target)
    ##填充关于encoder编码的三个结果
    max_length = max(x[1].shape[0] for x in batch)
    pad_target = torch.ones([batch_size, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][1]
        pad_target[sample_idx, : data.shape[0]] = data
    output.append(pad_target)
    #填充OOV_con_result、mask_query、Taboo
    max_length_oov = max(x[2].shape[0] for x in batch)
    max_OOV_length = max(x[2].shape[1] for x in batch)
    pad_target = torch.ones([batch_size, max_length_oov,max_OOV_length], dtype=torch.long) *-1
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][2]
        pad_target[sample_idx, : data.shape[0],: data.shape[1]] = data
    output.append(pad_target)
    #填充mask_query 原本就3维
    max_length=max(x[3].shape[0] for x in batch)
    pad_target = torch.zeros([batch_size, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][3]
        pad_target[sample_idx, : data.shape[0]] = data
    output.append(pad_target)
    #target_mask
    max_length_target = max(x[4].shape[1] for x in batch)
    label_len_target = max(x[4].shape[0] for x in batch)
    pad_target = torch.zeros([batch_size, label_len_target, max_length_target], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][4]
        pad_target[sample_idx, : data.shape[0], : data.shape[1]] = data
    output.append(pad_target)
    #tgt_seq_len
    tgt_number = max(x[5].shape[0] for x in batch)
    pad_target = torch.zeros([batch_size, tgt_number], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][5]
        pad_target[sample_idx, : data.shape[0]] = data
    output.append(pad_target)
    '''填充二维标签(用于多分类的标签)'''
    entit_labels_len1 = max(x[6].shape[0] for x in batch)
    entit_labels_len2 = max(x[6].shape[1] for x in batch)
    decoder_index_number = max(x[6].shape[2] for x in batch)
    pad_target = torch.zeros([batch_size, entit_labels_len1,entit_labels_len2,decoder_index_number], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][6]
        pad_target[sample_idx, : data.shape[0],: data.shape[1], : data.shape[2]] = data
    output.append(pad_target)

    max_bpe_len1 = max(x[7].shape[0] for x in batch)
    max_bpe_len2 = max(x[7].shape[1] for x in batch)
    max_bpe_len3 = max(x[7].shape[2] for x in batch)
    pad_target = torch.zeros([batch_size, max_bpe_len1,max_bpe_len2,max_bpe_len3], dtype=torch.long)
    for sample_idx in range(batch_size):  # 填充第一个
        data = batch[sample_idx][7]
        pad_target[sample_idx, : data.shape[0],: data.shape[1],: data.shape[2]] = data
    output.append(pad_target)


    output.append(torch.LongTensor([x[-4] for x in batch]))
    output.append([x[-3] for x in batch])##最后一个不需要填充
    output.append([x[-2] for x in batch])  ##最后一个不需要填充
    output.append([x[-1] for x in batch])##最后一个不需要填充
    return output
