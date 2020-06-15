#coding=utf-8
from typing import List, Any, Tuple, Union

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

PAD,CLS='[PAD]','[CLS]'
class DataProcess:
    def __init__(self,config):
        self.config=config

    def pad_sentence(self,path):
        sentences,label=self.get_data_label(path)
        print('处理数据中....')
        print('加上[CLS]并分词....')
        contents = []
        index = 0
        print('词到ids转换中...')
        for sent in tqdm(sentences):
            tokenized_sent=self.config.tokenizer.tokenize(sent)
            tokenized_sent =[CLS]+tokenized_sent
            token_ids=self.config.tokenizer.convert_tokens_to_ids(tokenized_sent)
            if len(token_ids)<self.config.pad_size:
                mask=[1]*len(token_ids)+[0]*(self.config.pad_size-len(token_ids))
                token_ids=token_ids+[0]*(self.config.pad_size-len(token_ids))
            else:
                mask=[1]*self.config.pad_size
                token_ids=token_ids[:self.config.pad_size]
            contents.append([np.array(token_ids),
                             np.array(int(label[index])),
                             np.array(mask)]
                            )
            index=index+1
        return contents

    def get_data_label(self,path):
        data_x,data_y=[],[]
        print('加载数据中.....')
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in tqdm(lines):
                line=line.strip().split('\t')
                data_x.append(line[0])
                data_y.append(line[1])
            f.close()
        return data_x,data_y

class TrainData(Dataset):
    def __init__(self,config):
        self.config=config
        self.data_process=DataProcess(self.config)
        self.train_data= self.data_process.pad_sentence(self.config.train_path)
        print(len(self.train_data))
    def __getitem__(self, index):
        data_X = self.train_data[index][0]
        data_y = self.train_data[index][1]
        mask = self.train_data[index][2]
        return data_X, data_y, mask
    def __len__(self):
        return len(self.train_data)

class DevData(Dataset):
    def __init__(self,config):
        self.config=config
        self.data_process=DataProcess(self.config)
        self.dev_data=self.data_process.pad_sentence(self.config.dev_path)
    def __getitem__(self, index):
        return self.dev_data[index]
    def __len__(self):
        return len(self.dev_data)

class TestData(Dataset):
    def __init__(self,config):
        self.config=config
        self.data_process=DataProcess(self.config)
        self.test_data= self.data_process.pad_sentence(self.config.test_path)
    def __getitem__(self, index):
        return self.test_data[index]
    def __len__(self):
        return len(self.test_data)
