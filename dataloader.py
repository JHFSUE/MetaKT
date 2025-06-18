# -*- coding: utf-8 -*-
import torch
from data.readdata import DataReader
import torch.utils.data as Data
import random
import tqdm
import logging

class KnowledgeTrackingDataset:
    def __init__(self, name):
        self.name = name

logger = logging.getLogger('main.dataloader')
# 准备N个不同的预处理好的知识追踪数据集
train_datasets = [
   # KnowledgeTrackingDataset("../dataset/assist2009/assist2009-110.txt"),
   # KnowledgeTrackingDataset("../dataset/assist2015/assist2015-99.txt"),
   # KnowledgeTrackingDataset("../dataset/assist2017/assist2017-101.txt"),
   # KnowledgeTrackingDataset("../dataset/Junyi/junyi1326.txt"),
   # KnowledgeTrackingDataset("../dataset/EdNet/Ednet1792.txt"),
   # KnowledgeTrackingDataset("../dataset/assist2012/assist2012-197.txt"),
   KnowledgeTrackingDataset("../dataset/algebra05/algebra05-137.txt"),
]
test_dataset = '../dataset/statics2011/static2011-5-1220.txt'
# test_dataset = '../dataset/assist2009/assist2009-5-110.txt'
# test_dataset = '../dataset/assist2015/assist2015-5-99.txt'
# test_dataset = '../dataset/assist2017/assist2017-5-101.txt'
# test_dataset = '../dataset/Junyi/junyi-5-1326.txt'
# test_dataset = '../dataset/EdNet/Ednet1792-5-1706.txt'
# test_dataset = '../dataset/assist2012/assist2012-5-193.txt'

class DataLoaderWrapper:
    def __init__(self, path, max_step, num_of_questions):
        self.handle = DataReader(path, max_step, num_of_questions)

    # 获取随机选择的支持集和查询集
    def get_data(self):
        return self.handle.getTrainData()

    def get_data_loader(self, batch_size):
        # 调用get——data,从readdata中获取随机选择的支持集和查询集
        support_set, query_set = self.get_data()

        t_support = torch.tensor(support_set.astype(float).tolist(), dtype=torch.float32)
        t_query = torch.tensor(query_set.astype(float).tolist(), dtype=torch.float32)

        support = Data.DataLoader(t_support, batch_size=batch_size, shuffle=True)
        query = Data.DataLoader(t_query, batch_size=batch_size, shuffle=True)
        # 释放内存，清除数据变量
        del support_set
        del query_set

        return support, query


def get_task_loader(datasets, question, length, batch_size):

    selected_dataset = random.choice(datasets)
    path = selected_dataset.name
    logger.info('train_path: ' + path)
    print('train_path: ' + path)

    loader_wrapper = DataLoaderWrapper(path, length, question)
    support_loader, query_loader = loader_wrapper.get_data_loader(batch_size)

    return support_loader, query_loader

# 获取训练任务
def get_train_task(question, length, batch_size):
    return get_task_loader(train_datasets, question, length, batch_size)


def get_test_path():
    return test_dataset

def getDataLoader(num_of_questions, max_step, path):
    handle = DataReader(path, max_step, num_of_questions)
    data = handle.getTotalData()
    return data
