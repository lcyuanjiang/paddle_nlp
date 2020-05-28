from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import io
import sys
import random
import numpy as np
import os
from paddle import fluid


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def data_reader(file_path, word_dict, num_examples, phrase, epoch, max_seq_len):
    """
    Convert word sequence into slot
    """
    unk_id = word_dict.get('<unk>')
    pad_id = 0
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith('label'):
                continue
            cols = line.strip().split("\t")
            if len(cols) != 2:
                sys.stderr.write("[NOTICE] Error Format Line!")
                continue
            label = int(cols[0])
            wids = [word_dict[x] if x in word_dict else unk_id
                    for x in cols[1].split(" ")]
            seq_len = len(wids)
            if seq_len < max_seq_len:
                for i in range(max_seq_len - seq_len):
                    wids.append(pad_id)
            else:
                wids = wids[:max_seq_len]
                seq_len = max_seq_len
            all_data.append((wids, label, seq_len))

    if phrase == "train":
        random.shuffle(all_data)

    num_examples[phrase] = len(all_data)
        
    def reader():
        """
        Reader Function
        """
        for epoch_index in range(epoch):
            for doc, label, seq_len in all_data:
                yield doc, label, seq_len
    return reader

class SentaProcessor(object):
    """
    Processor class for data convertors for senta
    """

    def __init__(self,
                 data_dir,
                 vocab_path,
                 random_seed,
                 max_seq_len):
        self.data_dir = data_dir
        self.vocab = load_vocab(vocab_path)
        self.num_examples = {"train": -1, "dev": -1, "infer": -1}
        np.random.seed(random_seed)
        self.max_seq_len = max_seq_len

    def get_train_examples(self, data_dir, epoch, max_seq_len):
        """
        Load training examples
        """
        return data_reader((self.data_dir + "/train.tsv"), self.vocab, self.num_examples, "train", epoch, max_seq_len)

    def get_dev_examples(self, data_dir, epoch, max_seq_len):
        """
        Load dev examples
        """
        return data_reader((self.data_dir + "/dev.tsv"), self.vocab, self.num_examples, "dev", epoch, max_seq_len)

    def get_test_examples(self, data_dir, epoch, max_seq_len):
        """
        Load test examples
        """
        return data_reader((self.data_dir + "/test.tsv"), self.vocab, self.num_examples, "infer", epoch, max_seq_len)

    def get_labels(self):
        """
        Return Labels
        """
        return ["0", "1","2"]

    def get_num_examples(self, phase):
        """
        Return num of examples in train, dev, test set
        """
        if phase not in ['train', 'dev', 'infer']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """
        Get train progress
        """
        return self.current_train_example, self.current_train_epoch

    def data_generator(self, batch_size, phase='train', epoch=1, shuffle=True):
        """
        Generate data for train, dev or infer
        """
        if phase == "train":
            return fluid.io.batch(self.get_train_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
            #return self.get_train_examples(self.data_dir, epoch, self.max_seq_len)
        elif phase == "dev":
            return fluid.io.batch(self.get_dev_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
        elif phase == "infer":
            return fluid.io.batch(self.get_test_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")


if __name__ == '__main__':
    dev_path='data/dev.tsv'
    word_dict=load_vocab('data/vocab.txt')
    num_examples={'train':-1,'dev':-1,'test':-1}
    reader=data_reader(dev_path,word_dict,num_examples,'dev',1,24)
    print(next(reader))