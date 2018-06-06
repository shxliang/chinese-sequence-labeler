import math
import random
from collections import OrderedDict


class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        # batch个数
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        """
        对输入的样本集划分batch，并在每个batch中进行padding
        :param data: 包含字序列、字index序列、分词特征序列、tag index序列
        :param batch_size: 
        :return: 
            batch_data: 每个元素为一个batch，一个batch有若干个样本，一个样本包含padding后的字序列、字index序列、分词特征序列、tag index序列
        """
        # 计算一个epoch有几个batch
        num_batch = int(math.ceil(len(data) / batch_size))
        # 以序列长度升序排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        # 这里只进行了排序，不先进行shuffle，目的是使同一batch中序列长度差别不大
        # 对同一batch里的序列使用0进行padding，padding到当前batch中最长序列长度
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        """
        对同一batch的数据进行padding
        :param data: 
        :return: 
            string：list，字序列
            chars：list，字index序列
            segs：list，分词特征序列
            features：list of dict，其他特征序列
            targets：目标序列
        """
        strings = []
        chars = []
        segs = []
        features = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])

        for line in data:
            string, char, seg, feature, target = line
            padding = [0] * (max_length - len(string))

            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)

            new_feature = OrderedDict()
            for feature_i, old_feature_i_inputs in feature.items():
                if feature_i == "0":
                    continue
                new_feature[feature_i] = old_feature_i_inputs + padding
            features.append(new_feature)

            targets.append(target + padding)
        return [strings, chars, segs, features, targets]

    def iter_batch(self, shuffle=False):
        # 是否对batch顺序进行shuffle
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]