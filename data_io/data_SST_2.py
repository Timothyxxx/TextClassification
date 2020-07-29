import csv
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


class DataIOSST2(object):
    def __init__(self, args):
        self.args = args

    def read_train_dev_test(self):
        train_word, train_label = self.read_data(self.args.data_dir + '/train.tsv')
        dev_word, dev_label = self.read_data(self.args.data_dir + '/dev.tsv')
        test_word = self.read_test(self.args.data_dir + '/test.tsv')
        return train_word, train_label, dev_word, dev_label, test_word

    @staticmethod
    def read_data(path):
        data = []
        label = []
        csv.register_dialect('my', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(path) as tsvfile:
            file_list = csv.reader(tsvfile, "my")
            first = True
            for line in file_list:
                if first:
                    first = False
                    continue
                data.append(line[0].strip().split(" "))
                label.append(line[1])
        csv.unregister_dialect('my')
        return data, label

    @staticmethod
    def read_test(path):
        data = []
        csv.register_dialect('my', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(path) as tsvfile:
            file_list = csv.reader(tsvfile, "my")
            first = True
            for line in file_list:
                if first:
                    first = False
                    continue
                data.append(line[1].strip().split(" "))
        csv.unregister_dialect('my')
        return data

    def get_data_loader(self, name):
        train_word, train_label, dev_word, dev_label, test_word = self.read_train_dev_test()
        return (DataLoader(TorchDataset(train_word, train_label), batch_size=self.args.batch_size,
                           shuffle=True, collate_fn=self.__collate_fn),
                DataLoader(TorchDataset(dev_word, dev_label), batch_size=self.args.batch_size, shuffle=True,
                           collate_fn=self.__collate_fn))

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch

    @staticmethod
    def add_padding(texts, items=None, digital=False):
        """
        Sorting by the length and add padding to the texts and the items(items could be the y values or else)

        :param texts: a list of different lists which need to pad
        :param items: require the 'list of tuple' type input(like '[(y, false),...]', y means the item with texts which need to
        change order together with the text ,and the false which is in the position of 'require' means )
        :param digital: the padding element'type (True means '0', False means '<PAD>')
        :return:
        """
        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_items = [], [], None
        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            if digital:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
            else:
                trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

            # This required specific if padding after sorting.
            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if digital:
                            item[-1].extend([0] * (max_len - len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        if items is not None:
            return trans_texts, trans_items, seq_lens
        else:
            return trans_texts, seq_lens


class TorchDataset(Dataset):
    def __init__(self, word, label):
        self.word = word
        self.label = label

    def __getitem__(self, item):
        return self.word[item], self.label[item]

    def __len__(self):
        return len(self.word)
