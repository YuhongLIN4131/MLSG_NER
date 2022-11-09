# encoding: utf-8


import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
import random
import math
import copy
import tqdm
import numpy as np
import os
import random
from itertools import chain
import torch
import itertools
import scipy.sparse as sp
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from scipy import stats
import time

from torch.utils.data import Dataset, DataLoader, BatchSampler


# seed_num = 666
# random.seed(seed_num)
# os.environ['PYTHONHASHSEED'] = str(seed_num)
# np.random.seed(seed_num)
# torch.manual_seed(seed_num)
# torch.cuda.manual_seed(seed_num)
# torch.cuda.manual_seed_all(seed_num)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

class MRCNERDataset(Dataset):  # 改进，让它能同时处理两种任务的数据
    """
    MRC NER Dataset and MLM task
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """

    def __init__(self, task_data, tokenizer: BertWordPieceTokenizer, max_length: int = 128,
                 pad_to_maxlen=False, prefix="train", task_id=6, label2token=None,  # 每个标签对应的token
                 Negative_sampling=1, part_entity_Negative_sampling=None,
                 target_type='word', use_part_label=True, use_part_entity=False, Discontinue=False,
                 non_entity_label=True,
                 ):  # 默认是NER任务
        self.all_data = task_data
        self.result_all_data = self.all_data
        self.max_length = max_length  #
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.task_id = task_id
        self.part_entity_Negative_sampling = part_entity_Negative_sampling  # 部分实体负采样的比例
        self.pad_to_maxlen = pad_to_maxlen  # fasle
        self.Negative_sampling = Negative_sampling
        self.label2token = label2token
        self.use_part_label = use_part_label
        self.target_type = target_type
        self.Discontinue = Discontinue
        self.use_part_entity = use_part_entity
        self.non_entity_label = non_entity_label  # 非实体和部分实体是否需要1的结束符号
        self.label_lengths = len(self.label2token)

        '''增加n个新的标签'''

    def get_task_id(self):
        return self.task_coding

    def __len__(self):
        return len(self.result_all_data)

    def calculate1(self, item):
        task_id = self.task_id
        context = item["context"].split()
        now_max_len = 128
        '''1、得到实体字位置'''
        new_entity_list = []  # 实体包含的字
        new_entity_index_list = []
        label_idx = []
        sample_id = item['sample_idx']
        if task_id >= 5:
            con_positin = item["start_end_con"]
            discon_positin = item["start_end_discon"]
            entity_label_discon = item["entity_label_discon"]
            entity_label_con = item["entity_label_con"]
            if len(con_positin) > 0:
                for isz, iz in enumerate(con_positin):
                    if iz[0] >= now_max_len or iz[-1] > now_max_len + 12:
                        continue
                    new_entity_list.append(context[iz[0]:iz[1] + 1])
                    new_entity_index_list.append([iz[0], iz[1]])
                    label_idx.append(entity_label_con[isz])
            if len(discon_positin) > 0:
                for isz, iz in enumerate(discon_positin):
                    if iz[0] >= now_max_len or iz[-1] > now_max_len + 12:
                        continue
                    temp = []
                    for i in range(len(iz) // 2):
                        temp.append(context[iz[2 * i]:iz[2 * i + 1] + 1])
                    new_entity_list.append(list(chain(*temp)))
                    new_entity_index_list.append([idx for idx in iz])
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
                    new_entity_index_list.append([start_position[isz], end_position[isz]])
                    label_idx.append(entity_label[isz])
        if len(new_entity_index_list) > 0:  ##截断操作
            now_max_len = max(now_max_len, max(list(chain(*new_entity_index_list))) + 1)
        sentence = context[0:now_max_len]
        '''给样本排序，前面很大的问题是因为训练的过程没有排序。
        1、按照起点位置排序，优先度为起>终'''
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
        '''2、按照终点顺序排序，即终点》起点'''
        '''让最长长度不能超过168'''

        now_ins = {}
        now_ins['entities'] = new_entity_list  # 每个span的位置
        now_ins['entity_tags'] = label_idx  # 样本的标签
        now_ins['task_id'] = task_id  # 样本的标签
        now_ins['type'] = item['type']  # 样本的标签
        now_ins['sample_id'] = sample_id  # 样本的标签
        now_ins['raw_words'] = sentence  # 以空格隔开的words，，加上了query
        now_ins['entity_spans'] = new_entity_index_list  # 每个实体的起点-终点对，非连续有多个，连续只有一个 (和前面不一样，终点就是终点，并不是终点的下一个位置)
        return now_ins

    def tokenizes(self, word):
        bpes1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True))
        return bpes1

    def __getitem__(self, item):  # 通过索引访问类中元素的时候会用到
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        '''
            这里讲一下处理格式《s》 实体类型 具体实体 《/s》终止
        '''
        '''
            使用query的方式进一步解决：
            1、先测试随机初始化10个单词

        '''
        target_shift = 2 + self.label_lengths  # 是由于第一位是sos，紧接着是eos, 然后是"<,>",是一个实体分割符
        if self.use_part_label:
            real_label_start = 2 + self.label_lengths - 1  # 严格标签的起点位置
        else:
            real_label_start = 2 + self.label_lengths
        ins = self.calculate1(self.result_all_data[item])
        # ins = self.result_all_data[item]
        '''把不连续实体的数据转成一样，需要加上label标签'''
        raw_words = ins['raw_words']
        task_id = ins["task_id"]
        sample_id = ins["sample_id"]
        word_bpes = [[self.tokenizer.bos_token_id]]
        first_OOV = []  # 用来取每个word第一个bpe
        OOV_con = [[0]]  # 存放OOV词的位置
        cur_bpe_len = 1
        '''添加分号  :  为分隔符'''
        start_context = cur_bpe_len
        max_OOV_len = 0
        for word in raw_words:
            bpes = self.tokenizes(word)  # 先分词再转id
            first_OOV.append(cur_bpe_len)
            OOV_con.append(list(range(cur_bpe_len, cur_bpe_len + len(bpes))))
            cur_bpe_len += len(bpes)  ##这个单词分词的长度
            max_OOV_len = max(max_OOV_len, len(bpes))
            word_bpes.append(bpes)
        word_bpes.append([self.tokenizer.eos_token_id])  # 结束符号
        OOV_con.append([cur_bpe_len])
        # first记录每一个分词的头位置
        assert len(first_OOV) == len(raw_words) == len(word_bpes) - 2
        lens = list(map(len, word_bpes))  # word_bpes中每个元素的长度
        cum_lens = np.cumsum(lens).tolist()  # np.cumsum累加函数
        '''entity_spans中终点位置并没有加一'''
        entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]#每个实体的起点-终点对，非连续有多个，连续只有一个
        label_idx = ins['entity_tags']  ##样本标记 1、2、3
        entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]#实体的位置
        target = [0]  # 特殊的sos  第一个起点词
        pairs = []
        '''添加start_endlabel'''
        _word_bpes = list(chain(*word_bpes))  # 整个句子的分词token
        for idx, entity in enumerate(entity_spans):
            cur_pair = []
            num_ent = len(entity) // 2
            for i in range(num_ent):  ##针对非连续的处理，非连续有多个这种起终
                start = entity[2 * i]
                end = entity[2 * i + 1]
                cur_pair_ = []
                if self.target_type == 'word':
                    cur_pair_.extend([cum_lens[k] for k in list(range(start, end + 1))])
                elif self.target_type == 'span':
                    cur_pair_.append(cum_lens[start])
                    cur_pair_.append(
                        cum_lens[end + 1] - 1)  # it is more reasonable to use ``cur_pair_.append(cum_lens[end-1])``
                elif self.target_type == 'bpe':
                    cur_pair_.extend(list(range(cum_lens[start], cum_lens[end + 1])))
                else:
                    raise RuntimeError("Not support other tagging")
                cur_pair.extend([p + target_shift for p in cur_pair_])  # extend就是追加的意思，相当于+
            assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])
            if self.target_type == "word":  # '''转化为整词的位置'''
                for ifd in range(len(cur_pair)):
                    for jd in range(len(OOV_con)):
                        if cur_pair[ifd] - target_shift in OOV_con[jd]:
                            cur_pair[ifd] = jd + target_shift
                            break
            cur_pair.append(label_idx[idx] + 1)  # 全实体标签
            pairs.append([p for p in cur_pair])
        target.extend(list(chain(*pairs)))
        target.append(1)  # 结尾符号
        '''实体的具体位置'''
        word_bpes = list(chain(*word_bpes))
        if self.target_type == "word":
            mask_query = [0] + [1] + (target_shift - 2) * [0] + [0] + (len(OOV_con) - 2) * [1] + [
                0]  # 不可能生成标签，所以标签的mask为0
        else:  # bpe范式
            mask_query = [0] + [1] + (target_shift - 2) * [0] + [0] + (len(word_bpes) - 2) * [1] + [0]
        assert len(word_bpes) < 500
        OOV_con_result = []
        '''对于bpe模式需要给出那些token是实体的起点token'''
        bpe_first = [l[0] + target_shift for l in OOV_con]
        bpe_tail = [l[-1] + target_shift for l in OOV_con]
        OOV_dict = {}  # 每个值对应的序列
        for i in OOV_con:
            OOV_dict[i[0]] = i
        for its in OOV_con:
            temp = max_OOV_len * [-1]
            temp[0:len(its)] = its
            OOV_con_result.append(copy.deepcopy(temp))
        '''先把数组送到torch'''
        word_bpes = torch.LongTensor(word_bpes)  # 放在设备2
        mask_query = torch.LongTensor(mask_query)  # 放在设备2
        OOV_con = torch.LongTensor(OOV_con_result)
        '''给其中的非头分词标注为 1'''
        bpe_tail_flag = []
        for ik in range(len(mask_query)):
            if ik < target_shift + 1:
                bpe_tail_flag.append(0)
            elif ik in bpe_tail:
                bpe_tail_flag.append(0)
            else:
                bpe_tail_flag.append(1)
        '''给其中的非尾分词标注为 1'''
        bpe_head_flag = []
        for ik in range(len(mask_query)):
            if ik < target_shift + 1:
                bpe_head_flag.append(0)
            elif ik in bpe_first:
                bpe_head_flag.append(0)
            else:
                bpe_head_flag.append(1)
        '''对于实体首词，直接接结束符，不能是0,1,2,以及标签字符，默认这里使用分词合并'''
        all_word_entity = []
        all_labels = set()  # 记录出现了那些标签
        '''获得每个词的下一词有哪些'''
        if self.prefix == "train":
            for i in range(target_shift + 1, len(mask_query) - 1):  # 最后一个点不能训练的
                if self.target_type == 'bpe' and i not in bpe_first:  # 是头分词才能开始运算
                    continue
                else:
                    temp_entity = []  #
                    for ids, j in enumerate(pairs):
                        if j[0] == i:  # 确实是实体的起点，一个起点可能对应多个起点，所以不退出
                            if [0] + j not in temp_entity:  # 去掉重复
                                temp_entity.append([0] + j[-1:] + j[0:-1] + [1])  # 没必要接终点符号
                                all_labels.add(j[-1])
                    if len(temp_entity) > 0:  # 这个位置对应的实体
                        for ki in temp_entity:
                            if ki not in all_word_entity:  # 去重
                                all_word_entity.append(ki)
            # 需要补充每个标签的位置:
            for la in range(2, 2 + self.label_lengths):
                if la not in all_labels:
                    all_word_entity.append([0, la, 1])
        else:  # 测试的时候,以每种类型实体作为起点即可
            for la in range(2, 2 + self.label_lengths):
                all_word_entity.append([0, la])
        all_word_entity_label = []  # 所有的标签
        # 计算标签，从具体实体标签后计算标签
        for k_po, i_en in enumerate(all_word_entity):
            labelNow = []
            for po, j_token in enumerate(i_en[0:-1]):
                Initial = [0] * len(mask_query)  # 初始化的标签
                '''可能就存在多标签的问题了'''
                Initial[i_en[po + 1]] = 1  # 首先，放置它应该有的标签
                for k_po2, i_en2 in enumerate(all_word_entity):  # 是否是多标签，多个可能
                    if k_po2 != k_po and i_en2[0:po + 1] == i_en[0:po + 1] and i_en[po + 1] != i_en2[po + 1]:
                        Initial[i_en2[po + 1]] = 1  # 存在的可能多标签
                labelNow.append(copy.deepcopy(Initial))
            all_word_entity_label.append(labelNow)
        '''##################'''
        '''接下来是给解码长度补齐'''
        target_max_len = max([len(ix) for ix in all_word_entity])  # 单词长度补齐
        new_all_target = []
        tgt_seq_len = []  # 记录生成块的
        target_mask = []  # 选择那些token会进行计算
        for ix in all_word_entity:
            tgt_seq_len.append(len(ix))
            temp = [1] * target_max_len
            temp[0:len(ix)] = ix
            new_all_target.append(copy.deepcopy(temp))  # 填充完的结果
            # 因为第一个点和 起点符号生成的值不需要学习
            target_mask.append([0] * 1 + [1] * (len(ix) - 1) + (target_max_len - len(ix)) * [0])  # 第一个点不需要学习
        '''记录每个tgt的长度'''
        max_mid_len = max([len(ix) for ix in all_word_entity_label])  # 每个句子中的标签数
        total_labels = []
        '''加一项，每个序列对应的标签长度'''
        for iz in all_word_entity_label:
            temp_labels = []
            for ki in range(max_mid_len):
                if ki >= len(iz):
                    now_label = [0] * len(mask_query)  # 填充点不需要标签
                else:
                    now_label = iz[ki]
                temp_labels.append(now_label)
            total_labels.append(copy.deepcopy(temp_labels))
        # 标注那些标签是填充的
        cal_label_mask = []
        for i in new_all_target:  # target_number,tgt_len
            temp_mask = []
            temp_mask.append([0] * len(mask_query))  # 第一个不学习
            for jPos, j in enumerate(i[1:-1]):
                if jPos == 0:  # 第一个字符,可以是所有单词
                    temp_mask.append(
                        [0] + [1] + [0] * (target_shift - 2) + [0] + [1] * (len(mask_query) - target_shift - 2) + [0])
                elif j == 1:  # 是填充的，只能是后序单词
                    temp_mask.append([0] * len(mask_query))
                else:  # 非填充，且是正式开始生成实体
                    temp_label = [0] + [1] + [0] * (target_shift - 2) + [0] * (len(mask_query) - target_shift)
                    if j > target_shift:
                        for jIndex in range(j + 1, len(temp_label) - 1):
                            temp_label[jIndex] = 1  # 下一个点参加训练
                    temp_mask.append(copy.deepcopy(temp_label))
            cal_label_mask.append(temp_mask)
        return [
            torch.LongTensor(new_all_target),  #
            word_bpes,
            OOV_con,
            mask_query,
            torch.LongTensor(target_mask),
            torch.LongTensor(tgt_seq_len),  # 输出长
            torch.LongTensor(total_labels),
            torch.LongTensor(cal_label_mask),
            len(word_bpes),  # 输入长
            pairs,
            sample_id,
            0 if ins["type"] == 'dev' else 1
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


class MultiTaskDataset(Dataset):  # 所有任务的数据集，一个单位为一个任务的数据集
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset  # 建立一个任务编号和任务数据相对应的词典

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)  # 它的长度是所有任务的所有行数

    def __getitem__(self, idx):  # 得到某个任务的具体那一行样本
        # print("我现在正在检查idx")
        # print(idx)
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, mix_opt='train'):
        self._datasets = datasets
        self._batch_size = batch_size
        self.mix_opt = mix_opt
        token_len = datasets.token_len  # 每个句子的token个数
        train_data_list = self._get_shuffled_index_batches(len(datasets), batch_size, token_len)  # dataset存的是所有任务的任务数据集
        self._train_data_list = train_data_list  # 这里每个batch中
        if self.mix_opt == "train":  # 训练的话需要打乱
            random.shuffle(self._train_data_list)

    def _get_shuffled_index_batches(self, dataset_len, batch_size, token_len, token_idmax=5000):  # dataset_len这个是指的行数，
        '''根据最大长度限制的采样'''
        i = 0
        index_batches = []
        while i < dataset_len:
            temp_batchsize = batch_size
            token_lenss = max(token_len[i:min(i + batch_size, dataset_len)]) * batch_size
            while token_lenss > token_idmax:  # 需要隔断
                temp_batchsize = temp_batchsize - 1
                token_lenss = max(token_len[i:min(i + temp_batchsize, dataset_len)]) * temp_batchsize
            index_batches = index_batches + [list(range(i, min(i + temp_batchsize, dataset_len)))]
            i = i + temp_batchsize
        if self.mix_opt == "train":  # 训练的话需要打乱
            random.shuffle(index_batches)
        return index_batches

    def __len__(self):  # 和mutildata输出是一样的大小
        return len(self._train_data_list)

    def __iter__(self):
        return iter(self._train_data_list)

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt):
        '''mix_opt主要用来判断第一个数据集是否需要被打乱： 0打乱，>0不打乱'''
        all_indices = []
        for i in range(0, len(train_data_list)):
            all_indices += [i] * len(train_data_list[i])
        all_indices += [0] * len(train_data_list[0])
        if mix_opt == "test":
            random.shuffle(all_indices)
        return all_indices  # 按照batch将所有的数据集都打乱


def run_dataset():
    """test dataset"""
    import os
    from datasets.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader
    # zh datasets
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    # # json_path = "/mnt/mrc/zh_onto4/mrc-ner.train"
    # is_chinese = True

    # en datasets
    bert_path = "/mnt/mrc/bert-base-uncased"
    json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(
                *batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("=" * 20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end + 1]))


if __name__ == '__main__':
    run_dataset()
