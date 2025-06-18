# -*- coding: utf-8 -*-

import numpy as np
import itertools
import tqdm



class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for length, ques, ans in tqdm.tqdm(itertools.zip_longest(*[file] * 3)):
                length = int(length.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]

                if length >= 2:
                    slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                    for i in range(slices):
                        temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                        steps = min(length, self.maxstep)
                        for j in range(steps):
                            index = i * self.maxstep + j
                            if ans[index] == 1:
                                temp[j][ques[index]] = 1
                            else:
                                temp[j][ques[index] + self.numofques] = 1
                        length -= steps
                        data.append(temp.tolist())

        data = np.array(data)
        # 打乱数据集
        np.random.shuffle(data)
        print('done: ' + str(data.shape))

        return data



    def getTrainData(self):
        print('loading Meta-Train data...')

        data = np.array(self.getData(self.path))
        # Randomly select 5 samples for the support set and 15 samples for the query set
        # 使用了 np.random.choice 从数据集中随机选择 5 个索引作为支持集。
        # 然后，使用 np.setdiff1d 从所有索引中排除支持集的索引，以得到查询集的索引
        all_indices = np.arange(len(data))
        support_indices = np.random.choice(all_indices, 256, replace=False)
        query_indices = np.random.choice(all_indices, 256, replace=False)
        support_set = np.array(data)[support_indices]
        query_set = np.array(data)[query_indices]
        return support_set, query_set

    # 获取五折交叉验证数据集
    def getTotalData(self):
        print('loading Mete-test data...')
        Data = np.array(self.getData(self.path))
        return Data
