import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
import random
import math
import copy
import scipy.sparse as sp
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from scipy import stats
from torch.utils.data import Dataset, DataLoader, BatchSampler

def data_arguments(all_data):  ##数据增强
    result = []
    for item in all_data:
        uncased = []
        task_id = item['task_id']
        for i in range(0, 10):
            uncased.append("[uncased" + str(i) + "]")
        query = uncased
        context = item["context"].split()
        now_max_len = self.max_length
        '''1、得到实体字位置'''
        new_entity_list = []  # 实体包含的字
        new_entity_index_list = []
        label_idx = []
        if task_id >= 6:
            con_positin = item["start_end_con"]
            discon_positin = item["start_end_discon"]
            entity_label_discon = item["entity_label_discon"]
            entity_label_con = item["entity_label_con"]
            if len(con_positin) > 0:
                for isz, iz in enumerate(con_positin):
                    if iz[0] >= now_max_len or iz[-1] > now_max_len + 12:
                        continue
                    new_entity_list.append(context[iz[0]:iz[1] + 1])
                    new_entity_index_list.append([iz[0] + len(query), iz[1] + len(query)])
                    label_idx.append(entity_label_con[isz])
            if len(discon_positin) > 0:
                for isz, iz in enumerate(discon_positin):
                    if iz[0] >= now_max_len or iz[-1] > now_max_len + 12:
                        continue
                    temp = []
                    for i in range(len(iz) // 2):
                        temp.append(context[iz[2 * i]:iz[2 * i + 1] + 1])
                    new_entity_list.append(list(chain(*temp)))
                    new_entity_index_list.append([idx + len(query) for idx in iz])
                    label_idx.append(entity_label_discon[isz])
        else:  # 连续实体
            start_position = item["start_position"]
            end_position = item["end_position"]
            entity_label = item["entity_label"]
            if len(start_position) > 0:
                for isz, iz in enumerate(start_position):
                    if start_position[isz] >= now_max_len or end_position[isz] > now_max_len + 12:
                        continue
                    new_entity_list.append(context[start_position[isz]:end_position[isz] + 1])
                    new_entity_index_list.append([start_position[isz] + len(query), end_position[isz] + len(query)])
                    label_idx.append(entity_label[isz])

        if len(new_entity_index_list) > 0:  ##截断操作
            now_max_len = max(now_max_len, max(list(chain(*new_entity_index_list))) - len(query) + 1)
        sentence = context[0:now_max_len]
        '''给样本排序，前面很大的问题是因为训练的过程没有排序'''
        for i in range(1, len(new_entity_index_list)):
            for j in range(0, len(new_entity_index_list) - i):
                if new_entity_index_list[j][0] > new_entity_index_list[j + 1][0]:
                    new_entity_index_list[j], new_entity_index_list[j + 1] = new_entity_index_list[j + 1], \
                                                                             new_entity_index_list[j]
                    new_entity_list[j], new_entity_list[j + 1] = new_entity_list[j + 1], new_entity_list[j]
                    label_idx[j], label_idx[j + 1] = label_idx[j + 1], label_idx[j]
                elif new_entity_index_list[j][0] == new_entity_index_list[j + 1][0] and new_entity_index_list[j][
                    -1] > new_entity_index_list[j + 1][-1]:
                    new_entity_index_list[j], new_entity_index_list[j + 1] = new_entity_index_list[j + 1], \
                                                                             new_entity_index_list[j]
                    new_entity_list[j], new_entity_list[j + 1] = new_entity_list[j + 1], new_entity_list[j]
                    label_idx[j], label_idx[j + 1] = label_idx[j + 1], label_idx[j]
        '''一个在decoder阶段随机mask的数据增强'''
        # 先统计含有的实体起点个数，
        '''如果可行再研究新的随机mask （2个及以上）多个实体的结果'''
        start_token_number = []
        for i in new_entity_index_list:
            if i[0] not in start_token_number:
                start_token_number.append(i[0])
        Dimension = len(start_token_number)  ##实体总个数
        Dimension = min(Dimension, 5)  # 只能进行最大3的置换
        if (Dimension >= 1) and 'train' in task_Types:  # 存在实体#存在实体
            for co_num in range(1, Dimension + 1):
                Combination = []
                for i in itertools.combinations(start_token_number, co_num):
                    Combination.append(list(i))
                for Arrangement in Combination:  ##进行len（max_start) mask实体的操作
                    '''设置一个静止区间'''
                    Taboo = torch.LongTensor((len(query) + len(sentence)) * [1])
                    for seq_num in range(len(new_entity_index_list)):
                        if new_entity_index_list[seq_num][0] in Arrangement:  ##属于被禁忌的对象
                            if len(new_entity_index_list[seq_num]) == 2:  # 连续实体
                                Taboo[
                                new_entity_index_list[seq_num][0]:new_entity_index_list[seq_num][-1] + 1] = 0  # 禁忌
                            else:  ##存在多个区间
                                num_ent = len(new_entity_index_list[seq_num]) // 2
                                for ijk in range(num_ent):
                                    sttt = new_entity_index_list[seq_num][2 * ijk]
                                    ennn = new_entity_index_list[seq_num][2 * ijk + 1]
                                    Taboo[sttt:ennn + 1] = 0  # 禁忌
                    ##建立一个
                    random_new_entity_list = []
                    random_label_idx = []
                    random_new_entity_index_list = []
                    flag = True
                    max_lens = int(0.5 * len(sentence))
                    if (Taboo == 0).sum() > max_lens:  ##约束mask实体的最大长度， 15% or 20%的约束力,每次只增加0.3的样本
                        continue
                    for seq_num in range(len(new_entity_index_list)):
                        # 不在禁忌中重新排序
                        if len(new_entity_index_list[seq_num]) == 2:  ##连续实体
                            if (Taboo[new_entity_index_list[seq_num][0]:new_entity_index_list[seq_num][
                                                                            -1] + 1] == 0).sum() == 0:
                                random_new_entity_list.append(new_entity_list[seq_num])
                                random_label_idx.append(label_idx[seq_num])
                                random_new_entity_index_list.append(new_entity_index_list[seq_num])
                        if len(new_entity_index_list[seq_num]) > 2:  # 非连续实体
                            num_ent = len(new_entity_index_list[seq_num]) // 2
                            sum_discon = 0
                            for ijk in range(num_ent):
                                sttt = new_entity_index_list[seq_num][2 * ijk]
                                ennn = new_entity_index_list[seq_num][2 * ijk + 1]
                                sum_discon += (Taboo[sttt:ennn + 1] == 0).sum()  # 禁忌
                            if sum_discon == 0:  ##没有屏蔽点
                                random_new_entity_list.append(new_entity_list[seq_num])
                                random_label_idx.append(label_idx[seq_num])
                                random_new_entity_index_list.append(new_entity_index_list[seq_num])
                        if (Taboo[new_entity_index_list[seq_num][0]:new_entity_index_list[seq_num][
                                                                        -1] + 1] == 0).sum() > 0 and (
                                Taboo[new_entity_index_list[seq_num][0]:
                                new_entity_index_list[seq_num][-1] + 1] == 1).sum() > 0:
                            flag = False
                            break  ##这个方法可能存在噪音，不要
                    if flag:
                        Taboo = Taboo.numpy().tolist()
                        result.append({'entities': random_new_entity_list,
                                       "entity_tags": random_label_idx,
                                       "task_Types": task_Types,
                                       "task_id": task_id,
                                       "raw_words": sentence,
                                       "query": query,
                                       "Taboo": copy.deepcopy(Taboo),
                                       "entity_spans": random_new_entity_index_list,
                                       "arguments_data":True})
        '''无论怎样这里都会输出'''
        Taboo = (len(query) + len(sentence)) * [1]
        result.append({'entities': new_entity_list,
                       "entity_tags": label_idx,
                       "task_Types": task_Types,
                       "task_id": task_id,
                       "raw_words": sentence,
                       "query": query,
                       "Taboo": copy.deepcopy(Taboo),
                       "entity_spans": new_entity_index_list,
                       "arguments_data":False})
    return result