# encoding: utf-8
from itertools import permutations
import argparse
import os
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
from collections import namedtuple
from typing import Dict
import json
from itertools import chain
import pytorch_lightning as pl
import torch
import itertools
import torch.nn.functional as F
import random
import copy
import numpy as np
from fastNLP import seq_len_to_mask
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BartTokenizer
from torch.optim import SGD
from models.bart import BartSeq2SeqModel
from models.metrics import Seq2SeqSpanMetric
from transformers import AdamW
from models.generater import SequenceGeneratorModel
from datasets.mrc_ner_dataset import MultiTaskBatchSampler
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.mrc_ner_dataset import MultiTaskDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from utils.get_parser import get_parser
from utils.radom_seed import set_random_seed
import logging
from models.losses import Seq2SeqLoss
from loss.focal_loss import FocalLoss
from loss.LabelSmoothingLoss import LabelSmoothingLoss


seed_num = 666
random.seed(seed_num)
os.environ['PYTHONHASHSEED'] = str(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

TASK_ID2STRING={"Onto":0,"conll03":1,"ace2004":2,"ace2005":3,'genia':4,'cadec':5,'share2013':6,'share2014':7}
TASK_ID2_BARTLR={"Onto":1e-5,"conll03":1e-5,"ace2004":1e-5,"ace2005":1e-5,'genia':1e-5,'cadec':1e-5,'share2013':1e-5,'share2014':1e-5}
TASK_ID2_OTHERLR={"Onto":1e-3,"conll03":1e-3,"ace2004":1e-3,"ace2005":1e-3,'genia':1e-3,'cadec':1e-3,'share2013':1e-3,'share2014':1e-3}
TASK_ID2_epoch={"Onto":30,"conll03":30,"ace2004":30,"ace2005":30,'genia':30,'cadec':30,'share2013':30,'share2014':30}
TASK_ID2_warm_up={"Onto":0.1,"conll03":0.1,"ace2004":0.1,"ace2005":0.1,'genia':0.1,'cadec':0.1,'share2013':0.1,'share2014':0.1}
TASK_ID2_batch_size={"Onto":16,"conll03":16,"ace2004":16,"ace2005":8,'genia':16,'cadec':16,'share2013':16,'share2014':16}
TASK_ITEST_batch_size={"Onto":32,"conll03":32,"ace2004":48,"ace2005":32,'genia':16,'cadec':32,'share2013':32,'share2014':24}
Dis={"Onto":False,"conll03":False,"ace2004":False,"ace2005":False,'genia':False,'cadec':True,'share2013':True,'share2014':True}

Dataset_label_number = [18, 4, 7, 7, 5, 1, 1, 1]
class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir
        self.loss_type = self.args.loss_type
        self.optimizer = args.optimizer
        self.tokenizer = BartTokenizer.from_pretrained(self.bert_dir)
        '''对应需要的超参数'''
        self.use_decoder = args.use_decoder  # 是否使用encoder后的编码作为decoder输入
        self.OOV_Integrate = args.OOV_Integrate  # 是否使用encoder端的OOV分词整合来作为decoder端的输入   必须在use_decoder=True才能使用
        self.use_cat = args.use_cat  # 对encoder的输出编码和decoder的的输出编码之间是否使用cat来进行额外判断
        self.Negative_sampling = args.Negative_sampling  # 负采样的比例
        '''主要参数设置完毕'''
        '''增加label的特殊token,使用部分实体标签的话也只需要多增加一个token即可,全部加上那个部分实体标记'''
        self.label_num = Dataset_label_number[args.task_id]  # 实体的标签数量
        if self.args.use_part_entity==False:#不使用部分实体
            specific_token_file = self.args.data_dir + '/' + "specific_token.json"
        elif self.args.use_part_label:
            # specific_token_file = self.args.data_dir+'/'+"specific_token2.json"
            specific_token_file = self.args.data_dir+'/'+"specific_token3.json"#只有1个实体标签
        else:
            specific_token_file = self.args.data_dir+'/'+"specific_token.json"
        specific_token = json.load(open(specific_token_file, encoding="utf-8"))
        mapping_uncased=[]
        for i in specific_token:
            mapping_uncased.append(i)#加上
        self.tokenizer.add_tokens(mapping_uncased)  # 在字典中增加了新词汇
        # self.tokenizer.unique_no_split_tokens = self.tokenizer.unique_no_split_tokens + mapping_uncased
        '''更新词汇完毕，然后给标签对应token_id'''
        self.label2token = []
        for i in specific_token:
            self.label2token.append(self.tokenizer.convert_tokens_to_ids(i))
        '''接下来放置模型'''
        self.target_shift = 2 + self.label_num  # 有几个实体类型
        self.now_epoch=0
        self.metric = Seq2SeqSpanMetric()
        self.GetLoss=Seq2SeqLoss(loss_type=self.loss_type,dataset_name=self.args.dataset_name)
        length_penalty = 1
        self.model = BartSeq2SeqModel.build_model(self.bert_dir, self.tokenizer, use_decoder=self.use_decoder, use_cat=self.use_cat,
                                                  OOV_Integrate=self.OOV_Integrate,label_ids=self.label2token,
                                                  target_type=args.target_type,dataset_name=self.args.dataset_name,
                                                  label_Attend=args.label_Attend,use_biaffine1=args.use_biaffine1,
                                                  use_distance_embedding=args.use_distance_embedding)
        print("字典总长度：{}".format(self.model.decoder.decoder.embed_tokens.weight.data.size(0)))
        self.model = SequenceGeneratorModel(self.model, bos_token_id=args.bos_token_id,
                                       eos_token_id=args.eos_token_id,max_length=args.max_len,
                                       max_len_a=args.max_len_a, num_beams=args.num_beams, do_sample=False,
                                       repetition_penalty=1, length_penalty=length_penalty, pad_token_id=1,
                                       restricter=None,dataset_name=self.args.dataset_name,use_part_label=self.args.use_part_label,
                                        target_type=args.target_type)
        all_data_train = json.load(open(self.args.data_dir + "/entity.train", encoding="utf-8"))
        all_data_dev = json.load(open(self.args.data_dir + "/entity.dev", encoding="utf-8"))
        all_data_test = json.load(open(self.args.data_dir + "/entity.test", encoding="utf-8"))
        self.train_data_arguments = all_data_train
        self.dev_data = all_data_dev
        self.test_data = all_data_test
        self.dev_test_top3_score={}
        self.epochs=0
        self.Generateor={}#存储每个样本的可能负样本
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.3,
                            help="mrc dropout rate")
        parser.add_argument("--chinese", action="store_true",
                            help="is chinese dataset")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw",
                            help="loss type")
        parser.add_argument("--bos_token_id", type=int, default=0, help="解码的起点token")
        parser.add_argument("--eos_token_id", type=int, default=1, help="解码的终点token")
        parser.add_argument("--use_part_label", type=bool, default=True, help="是否为部分实体使用对应的部分实体编码")
        parser.add_argument("--use_distance_embedding", type=bool, default=True, help="是否使用距离编码")
        parser.add_argument("--use_biaffine1", type=bool, default=True, help="是否使用双反射代替点积")
        parser.add_argument("--label_Attend", type=bool, default=True, help="是否进行标签交互")

        parser.add_argument("--use_part_entity", type=bool, default=True, help="是否需要部分实体")
        parser.add_argument("--dice_smooth", type=float, default=1e-8, help="smooth value of dice loss")
        parser.add_argument("--num_beams", type=int, default=1, help="束搜索的宽度")
        parser.add_argument("--loss_type", choices=["cross_entropy", "LabelSmoothingLoss", "focal"], default="adamw",
                            help="损失类比")
        parser.add_argument("--target_type",choices=["word", "span", "bpe"], default="word", help="解码组装的类型")
        parser.add_argument("--max_len", type=int, default=10, help="解码出的最大长度")
        parser.add_argument("--task_id", type=int, default=6, help="当前处理的数据集名称")
        parser.add_argument("--dataset_name", type=str, default="share2013", help="数据集名称")
        parser.add_argument("--length_penalty", type=float, default=1.0, help="长度惩罚项")
        parser.add_argument("--part_entity_Negative_sampling", type=float, default=0.5, help="部分实体的负采样比例")
        parser.add_argument("--max_len_a", type=float, default=1.0, help="最终长度 = max_len+ src_len*max_len_a")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        return parser

    def tokenizes(self, word):
        bpes1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True))
        return bpes1
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        # '''下游模块和BART模块不公用相同学习率'''
        # bert_params = set(self.model.bert.parameters())
        # other_params = list(set(self.model.parameters()) - bert_params)
        # no_decay = ['bias', 'LayerNorm.weight']
        # params = [
        #     {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'lr': config.bert_learning_rate,
        #      'weight_decay': config.weight_decay},
        #     {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
        #      'lr': config.bert_learning_rate,
        #      'weight_decay': 0.0},
        #     {'params': other_params,
        #      'lr': config.learning_rate,
        #      'weight_decay': config.weight_decay},
        # ]
        bart1="seq2seq_model.decoder.decoder"
        bart2 ="seq2seq_model.encoder.bart_encoder"
        bert_parms=[]
        other_params=[]
        '''冻住一层，也即每个单词的基本意思不变，利于推广'''
        freeze_layers = "embed_tokens.weight"
        for name, param in self.model.named_parameters():
            # if freeze_layers in name:
            #     param.requires_grad = False
            if bart2 in name or bart1 in name:
                bert_parms.append([name,param])
            else:
                other_params.append(param)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in bert_parms if not any(nd in n for nd in no_decay)],#and freeze_layers not in n
                'lr':TASK_ID2_BARTLR[self.args.dataset_name],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in bert_parms if any(nd in n for nd in no_decay)],
                'lr': TASK_ID2_BARTLR[self.args.dataset_name],
                "weight_decay": 0.0,
            },
            {'params': other_params,
                 'lr': TASK_ID2_OTHERLR[self.args.dataset_name],
                 'weight_decay': self.args.weight_decay,},
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = round(t_total * TASK_ID2_warm_up[self.args.dataset_name])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        target, word_bpes,OOV_con,mask_query,target_mask,tgt_seq_len,all_word_entity_label,cal_label_mask,src_seq_len,target_span,sample_id,types = batch
        batch_size, target_number, entity_len, dimss = all_word_entity_label.size()
        outputs = self.model(word_bpes, target, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len,
                             mask_query=mask_query,OOV_con=OOV_con,all_word_entity_label=all_word_entity_label,
                             target_mask=target_mask,cal_label_mask=cal_label_mask)
        loss = self.GetLoss.get_loss(outputs["pred"],
                                     labels=outputs["labels"],
                                     cal_label_mask=outputs["cal_label_mask"])
        tf_board_logs[f"train_loss"] = loss
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}
        target, word_bpes,OOV_con,mask_query,target_mask,tgt_seq_len,all_word_entity_label,cal_label_mask,src_seq_len,target_span,sample_id,types = batch
        pred = self.model.predict(word_bpes, src_seq_len=src_seq_len,mask_query=mask_query,OOV_con=OOV_con,
                                     tgt_tokens=target, target_span=target_span)
        '''评价,'''
        fn, tp, fp = self.metric.evaluate(target_span, pred)
        """"""
        output['test_tp'] = 0
        output['test_fp'] = 0
        output['test_fn'] = 0
        output['dev_tp'] = 0
        output['dev_fp'] = 0
        output['dev_fn'] = 0

        for i in range(len(types)):
            if types[i]==0:#dev
                output['dev_tp'] += tp[i]
                output['dev_fp'] += fp[i]
                output['dev_fn'] += fn[i]
            else:#test
                output['test_tp'] += tp[i]
                output['test_fp'] += fp[i]
                output['test_fn'] += fn[i]
        return output

    def validation_epoch_end(self, outputs):
        """"""
        tensorboard_logs = {}
        '''先求验证集'''
        span_tp = torch.LongTensor([x['dev_tp'] for x in outputs]).sum()
        span_fp = torch.LongTensor([x['dev_fp'] for x in outputs]).sum()
        span_fn = torch.LongTensor([x['dev_fn'] for x in outputs]).sum()
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        print("验证集当前的精度：{}，召回：{}，F1：{}".format(span_precision,span_recall,span_f1))
        # # print("验证集当前的span_tp：{}，span_fp：{}，span_fn：{}".format(span_tp,span_fp,span_fn))
        # '''再求测试集'''
        test_span_tp = torch.LongTensor([x['test_tp'] for x in outputs]).sum()
        test_span_fp = torch.LongTensor([x['test_fp'] for x in outputs]).sum()
        test_span_fn = torch.LongTensor([x['test_fn'] for x in outputs]).sum()
        test_span_recall = test_span_tp / (test_span_tp + test_span_fn + 1e-10)
        test_span_precision = test_span_tp / (test_span_tp + test_span_fp + 1e-10)
        test_span_f1 = test_span_precision * test_span_recall * 2 / (test_span_recall + test_span_precision + 1e-10)
        print("测试集当前的精度：{}，召回：{}，F1：{}".format(test_span_precision, test_span_recall, test_span_f1))
        return {'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        if prefix=="train":
            task0 = self.train_data_arguments
        elif prefix=="dev":
            '''每个epoch处理一遍数据，得到生成的可能负样本'''
            task0 =self.test_data +self.dev_data
        else:
            task0 = self.test_data+self.dev_data

        dataset = MRCNERDataset(task_data=task0,
                                tokenizer=self.tokenizer,
                                max_length=self.args.max_length,
                                pad_to_maxlen=False,
                                prefix=prefix,
                                task_id=self.args.task_id,#属于哪一个任务
                                label2token =self.label2token,#每个标签对应的token
                                Negative_sampling = self.Negative_sampling,  # 负采样的比例
                                part_entity_Negative_sampling = self.args.part_entity_Negative_sampling,
                                target_type=self.args.target_type,
                                use_part_label=self.args.use_part_label,
                                use_part_entity=self.args.use_part_entity,
                                Discontinue=Dis[self.args.dataset_name],
                                non_entity_label=self.args.non_entity_label
                                )
        if limit is not None:
            dataset = TruncateDataset(dataset, limit)
        if prefix == "train":
            now_batch = self.args.batch_size
        else:
            now_batch = TASK_ITEST_batch_size[self.args.dataset_name]
        '''装载的时候写一个截断'''
        # samplers=MultiTaskBatchSampler(dataset,now_batch,mix_opt=prefix)#
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=now_batch,
            num_workers=self.args.workers,
            shuffle=False if prefix!='train' else True,
            # shuffle=False,
            # sampler=samplers,
            collate_fn=collate_to_max_length
        )
        return dataloader

def main():
    """main"""

    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    ##定义一下训练所需的超参数
    args.bert_config_dir="/home/wyq/BARTNER-main-acc/facebook/bart-large"
    # args.bert_config_dir="/home/wyq/BART/pytorch_model30"
    args.default_root_dir = "mnt/data/mrc/train_logs/debug"
    args.max_length = 320
    args.batch_size = 16
    args.gpus = "1"
    args.progress_bar_refresh_rate=10#每10步显示一次
    args.loss_type="focal" #"ASL","crossentropy"
    args.dataset_name = "ace2004"  #任务名"Onto":0,"conll03":1,"ace2004":3,"ace2005":4,'genia':5,'cadec':6,'share2013':7,'share2014':8
    args.data_dir = '/home/wyq/BARTNER-main/data/' + args.dataset_name
    args.task_id=TASK_ID2STRING[args.dataset_name]#任务id
    args.gradient_clip_val = 5.0#
    args.max_epochs=TASK_ID2_epoch[args.dataset_name]#
    args.lr = TASK_ID2_BARTLR[args.dataset_name]
    '''关于模型的超参数'''
    '''使用bpe算法，将cat换成一个额定的分类器'''
    args.target_type = "bpe"#word就是整合词编码，写一个bpe的方案
    args.use_decoder = True  # 是否使用encoder后的编码作为decoder输入   EAD
    args.OOV_Integrate = False  # 是否将oov进行池化
    args.use_cat = True  # 是否添加MLP层来增强
    args.use_part_label = False  # 是否为部分实体使用部分实体标签
    args.use_part_entity = False  # 是否使用部分实体的概念
    args.use_biaffine1 = True
    # 计算点积的时候是否换成biaffine
    args.label_Attend = True  # 标签之间是否进行进一步的交互
    args.use_distance_embedding = False  # 标签之间是否进行进一步的交互
    args.gnereate_from_NULL = True  # 从空序列开始生成实体（也即是从实体开始符直接生成）
    args.non_entity_label = True  # 非实体是否需要一个标签
    '''需求1：非实体词直接不接下一个词，
    需求2：多走一步，抑制暴露偏差问题'''
    args.Negative_sampling = 0# 对每个句子不在实体中的词进行0.5的概率负采样
    args.part_entity_Negative_sampling = 0 # 对部分实体也进行负采样，防止实体和部分实体之间比例的严重失衡
    args.num_beams=4#进行束搜索的宽度
    '''关于模型的参数都要在这里提前给出'''
    args.reload_dataloaders_every_epoch = True
    args.check_val_every_n_epoch = 1
    args.accumulate_grad_batches = TASK_ID2_batch_size[args.dataset_name]//args.batch_size
    args.val_check_interval = 1.0#0.25指每一个epoch验证4次
    args.num_sanity_val_steps = 0
    args.precision = 16
    '''结束'''
    args.num_beams = 4  # 进行束搜索的宽度
    model = BertLabeling(args)
    if args.pretrained_checkpoint:  ##是否读检查点
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=3,#前三验证
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )

    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()