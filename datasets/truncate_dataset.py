# encoding: utf-8

from torch.utils.data import Dataset


class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""
    def __init__(self, dataset: Dataset, max_num: int = 100):
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))#样本个数截断的意思

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)
