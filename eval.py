# -*- coding: utf-8 -*-

import tqdm
import torch
import logging

import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')


def auc_acc_recall_precision(gt, pred):
    # gt和pred都是PyTorch张量，大小为[batch_size, num_classes]
    # 其中，gt是真实标签，pred是模型预测
    # 需要将它们转换为一维张量，大小为[batch_size * num_classes]
    gt = gt.view(-1)
    pred = pred.view(-1)

    # 计算AUC
    fpr, tpr, _ = metrics.roc_curve(gt.detach().cpu().numpy(), pred.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    # 计算ACC
    threshold = 0.5
    binary_pred = (pred >= threshold).float()
    correct = torch.sum(binary_pred == gt).item()
    total = gt.numel()
    acc = correct / total

    # 计算recall和precision
    binary_pred = binary_pred.view(-1)
    tp = torch.sum((binary_pred == 1) & (gt == 1)).item()
    fn = torch.sum((binary_pred == 0) & (gt == 1)).item()
    fp = torch.sum((binary_pred == 1) & (gt == 0)).item()
    tn = torch.sum((binary_pred == 0) & (gt == 0)).item()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    logger.info('auc: ' + str(auc) +' acc: ' + str(acc) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision))
    print('auc: ' + str(auc) +' acc: ' + str(acc)+ ' recall: ' + str(recall) +
          ' precision: ' + str(precision))
    return auc, acc, recall, precision



class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device

    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:self.num_of_questions] + batch[student][:, self.num_of_questions:]  # shape: [length, questions]
            temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],dtype=torch.long, device=self.device)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] - batch[student][:, self.num_of_questions:]).sum(1) + 1) // 2)[1:]
            for i in range(len(p) - 1, -1, -1):
                if p[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth

def train_epoch(Maml_net, trainLoader, meta_test_optim, loss_func, device):
    """
      Train the model for one epoch.

      Args:
      - Maml_net: The neural network model to be trained
      - trainLoader: The DataLoader containing training data
      - meta_test_optim: The optimizer for updating model parameters
      - loss_func: The loss function for computing the training loss
      - device: Device for computation (CPU or GPU)

      Returns:
      - Maml_net: Updated model after meta-testing
      """
    Maml_net.to(device)
    Maml_net.train()
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        pred = Maml_net(batch)
        loss, prediction, ground_truth = loss_func(pred, batch)
        meta_test_optim.zero_grad()
        loss.backward()
        # 梯度截断，防止在RNNs或者LSTMs中梯度爆炸的问题
        torch.nn.utils.clip_grad_norm_(Maml_net.parameters(), max_norm=1.0, norm_type=2)
        meta_test_optim.step()
        # print(str(loss))
    return Maml_net

def test_epoch(model, testLoader, loss_func, device):
    model.to(device)
    model.eval()
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch)
        loss, p, a = loss_func(pred, batch)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        # return(performance(ground_truth, prediction))
    return (auc_acc_recall_precision(ground_truth, prediction))