import torch
import numpy as np
from data.dataloader import get_train_task
from data.dataloader import get_test_path
from data.dataloader import getDataLoader
import argparse
import time
from meta import Meta
from evaluation import eval
import logging
from datetime import datetime
import wandb
import random
from sklearn.model_selection import KFold
from copy import deepcopy
from torch import optim
import torch.utils.data as Data
from eval import test_epoch,train_epoch

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(model_name, log_directory='log'):
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)

    date = datetime.now()
    handler = logging.FileHandler(
        f'{log_directory}/{date.year}_{date.month}_{date.day}_{model_name}_result.log')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
# def login(args):
#      # 登录
#     wandb.login(key='f69473dfcc8c3ec2407645fbf13a4319ac17ed11')  # 密钥在主页
#     wandb.init(
#     project="全部数据集algebra05-5-136调优",  # 大框架是什么，最高一级
#     name="*MAML-dataset=assist2009-110 hidden=10 layer=1 epochs_1=200 epochs=40 support_indices=256 meta_lr=0.001 update_lr=0.1 bs=64 upadte_step=1",
#     # notes="",  # 一般是简单描述这次train你动了哪些部分，比如网络加了注意力机制等等
#     )
#     wandb.config.update(args)
def train_model(logger, maml, loss_func, train_spt, train_qry):
    maml.train()
    loss, model= maml(train_spt, train_qry, loss_func)
    logger.info("meta-train-loss_q: %f", loss)
    print("meta-train-loss_q: %f", loss)
    return model
def evaluate_model(Maml_net, testLoader, loss_func, device):
    auc, acc, recall, precision = test_epoch(Maml_net, testLoader, loss_func, device)
    return auc, acc, recall, precision
def main():
    # login(args)
    # 打印输出参数
    print(args)
    setup_seed(args.seed)
    # 在代码开始处记录开始时间
    start_time = time.time()
    # 假设你的模型名是 'my_model'
    logger = setup_logger(model_name= args.model)
    logger.info(args)


    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    maml = Meta(args, device)
    # 检测模型参数的梯度信息
    # wandb.watch(maml)
    num = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(maml)
    print('Total trainable tensors:', num)
    # 定义损失函数
    loss_func = eval.lossFunc(args.question, args.length, device)
    # 定义元训练轮数
    epochs = int(args.epochs)

    for epoch in range(epochs):
        logger.info(f'epoch {epoch + 1}')
        print('epoch: ' + str(epoch + 1) + '/' + str(epochs))
        # 元训练数据集         meta-train
        train_spt, train_qry = get_train_task(args.question, args.length, args.train_batch_size)
        # 训练
        updated_model = train_model(logger, maml, loss_func, train_spt, train_qry)

    # 五折交叉验证
    print("五折交叉验证")
    # 获取进行五折交叉验证的数据集路径
    path = get_test_path()
    # 获取交叉验证数据
    data = getDataLoader(args.question ,args.length, path)
    total_auc_list = []
    total_acc_list = []
    total_recall_list = []
    total_precision_list = []
    # 将数据集分为args['--fold']折
    kf = KFold(n_splits=int(args.fold), shuffle=True, random_state=3)
    # kf 是已经初始化的 KFold 对象，data 是数据集
    for fold, (train_indexes, test_indexes) in enumerate(kf.split(data)):
        # 创建模型和优化函数
        Maml_net = deepcopy(updated_model)
        print(Maml_net)
        meta_test_optim = optim.Adam(Maml_net.parameters(), lr=args.lr)
        auc_list = []
        acc_list = []
        recall_list = []
        precision_list = []
        print('**'*10+str(fold)+'**'*10)
        # 获取测试和训练集
        testData = data[test_indexes].tolist()
        trainData = data[train_indexes].tolist()
        train = np.array(trainData)
        test = np.array(testData)
        dtrain = torch.tensor(train.astype(float).tolist(), dtype=torch.float32)
        dtest = torch.tensor(test.astype(float).tolist(), dtype=torch.float32)

        trainLoader = Data.DataLoader(dtrain, batch_size=args.bs, shuffle=True)
        testLoader = Data.DataLoader(dtest, batch_size=args.bs, shuffle=False)
        best_auc = 0 #定义获取最好的结果
        # 元测试迭代
        for epoch in range(args.epochs_1):
            print('epoch: ' + str(epoch + 1) + '/' + str(args.epochs_1))
            # 训练
            Maml_net = train_epoch(Maml_net, trainLoader, meta_test_optim, loss_func, device)
            logger.info(f'epoch {epoch + 1}')
            # 测试
            auc, acc, recall, precision = evaluate_model(Maml_net, testLoader, loss_func, device)
            # 记录每个epoch的测试结果到WandB
            if fold == 0:
                test_log = {
                    "Fold_AUC": auc,
                    "Fold_Accuracy": acc,
                    "Fold_Recall": recall,
                    "Fold_Precision": precision
                }
                # wandb.log(test_log, step=epoch)
            # 记录所有数据
            total_auc_list.append(auc)
            total_acc_list.append(acc)
            total_recall_list.append(recall)
            total_precision_list.append(precision)
            # 记录每一折的数据
            auc_list.append(auc)
            acc_list.append(acc)
            recall_list.append(recall)
            precision_list.append(precision)

            if auc > best_auc:
                print('best checkpoint')
                logger.info('best checkpoint---auc:  ' + str(auc))
                # torch.save({'state_dict': model.state_dict()}, 'checkpoint/' + model_type + '.pth.tar')
                best_auc = auc

        logger.info('mean_auc: ' + str(np.mean(auc_list)) + ' mean_acc: ' + str(np.mean(acc_list))
                    + ' mean_recall: ' + str(np.mean(recall_list)) + ' mean_precision: ' + str(np.mean(precision_list)))
        print('mean_auc: ' + str(np.mean(auc_list)) + ' mean_acc: ' + str(np.mean(acc_list))
              + ' mean_recall: ' + str(np.mean(recall_list)) + ' mean_precision: ' + str(np.mean(precision_list)))
        logger.info('var_auc: ' + str(np.var(auc_list)) + ' var_acc: ' + str(np.var(acc_list))
                    + ' var_recall: ' + str(np.var(recall_list)) + ' var_precision: ' + str(np.var(precision_list)))
        print('var_auc: ' + str(np.var(auc_list)) + ' var_acc: ' + str(np.var(acc_list))
              + ' var_recall: ' + str(np.var(recall_list)) + ' var_precision: ' + str(np.var(precision_list)))

    logger.info('total_mean_auc: ' + str(np.mean(total_auc_list)) + ' total_mean_acc: ' + str(np.mean(total_acc_list))
                + ' total_mean_recall: ' + str(np.mean(total_recall_list)) + ' total_mean_precision: ' + str(
        np.mean(total_precision_list)))

    print('total_mean_auc: ' + str(np.mean(total_auc_list)) + ' total_mean_acc: ' + str(np.mean(total_acc_list))
          + ' total_mean_recall: ' + str(np.mean(total_recall_list)) + ' total_mean_precision: ' + str(
        np.mean(total_precision_list)))

    logger.info('total_var_auc: ' + str(np.var(total_auc_list)) + ' total_var_acc: ' + str(np.var(total_acc_list))
                + ' total_var_recall: ' + str(np.var(total_recall_list)) + ' total_var_precision: ' + str(
        np.var(total_precision_list)))

    print('total_var_auc: ' + str(np.var(total_auc_list)) + ' total_var_acc: ' + str(np.var(total_acc_list))
          + ' total_var_recall: ' + str(np.var(total_recall_list)) + ' total_var_precision: ' + str(
        np.var(total_precision_list)))
    # 在代码结束时释放资源
    logger.info('Cleaning up resources and ending the program.')
    print('total_mean_auc:       ' + str(np.mean(total_auc_list))+"+"+str(np.var(total_auc_list)))
    print('total_mean_acc:       ' + str(np.mean(total_acc_list))+"+"+str(np.var(total_acc_list)))
    print('total_mean_recall:    ' + str(np.mean(total_recall_list))+"+"+str(np.var(total_recall_list)))
    print('total_mean_precision: ' + str(np.mean(total_precision_list))+"+"+ str(np.var(total_precision_list)))
    # 如果你使用了GPU，确保释放CUDA内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # 关闭日志记录器
    logger.handlers[0].close()
    logger.propagate = False

    # 打印程序结束和总耗时
    end_time = time.time()
    time_elapsed = end_time - start_time

    minutes_elapsed = time_elapsed / 60  # 将秒转换为分钟

    logger.info('Total time elapsed: {:.2f} minutes'.format(minutes_elapsed))
    print('Total time elapsed: {:.2f} minutes'.format(minutes_elapsed))
    # 关闭WandB运行
    # wandb.finish()
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epochs', type=int, help='epoch number', default=5)
    argparser.add_argument('--epochs_1', type=int, help='test epoch number', default=100)
    argparser.add_argument('--hidden_size', type=int, help='task-level inner update steps', default=100)
    argparser.add_argument('--layer_size', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--length',type=int,help='max length of question sequence',default=5)
    argparser.add_argument('--question', type=int, help='max of question sequence num of question', default=1221)
    argparser.add_argument('--train_batch_size', type=int, help='meta train batch size', default=64)
    argparser.add_argument('--bs', type=int, help='meta test batch size', default=64)
    # 外循环元学习率
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    # 内循环基学习率
    argparser.add_argument('--update_lr', type=float, help='task-level inner upd ate learning rate', default=0.001)
    argparser.add_argument('--lr', type=float, help='task-level inner upd ate learning rate', default=0.001)

    # 内循环更新步数
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--model', type=str, help='model type', default='MAML-lstm')
    argparser.add_argument('--dropout', type=int, help='task-level inner update steps', default=0.1)
    argparser.add_argument('--seed', type=int, help='random seed', default=59)
    argparser.add_argument('--fold',type=int,help='K_fold',default=5)


    args = argparser.parse_args()

    main()
