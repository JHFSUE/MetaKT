import torch
from torch import nn
import tqdm
from torch import optim
from learner import LSTMModel
from copy import deepcopy

class Meta(nn.Module):
    """
        Meta Learner using Model-Agnostic Meta-Learning (MAML) approach.
    """
    def __init__(self, args, device):
        """
        Initialize Meta Learner
        Args:
            args: Arguments containing meta learning parameters
            device: Device for computation (CPU or GPU)
        """
        super(Meta, self).__init__()
        self.device = device
        self.update_lr = args.update_lr  # Learning rate for task-level updates
        self.meta_lr = args.meta_lr     # Learning rate for meta-level updates
        self.update_step = args.update_step  # Number of gradient updates for each task
        self.net = LSTMModel(args.question * 2, args.hidden_size, args.layer_size, args.question, self.device, args.dropout)
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    # meta-train
    def forward(self, x_spt, x_qry, loss_func):
        """
        Args:
            x_spt: Support set
            x_qry: Query set
            loss_func: Loss function for evaluation
        Returns:
            loss_q_1: Loss after meta-training
            self.net: Updated model after meta-training
        """
        # Copy the model for task-specific training
        Maml_net = deepcopy(self.net)
        Maml_net.to(self.device)
        Maml_net.train()
        losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i

        for k in tqdm.tqdm(range(self.update_step), desc='Meta-Training:    ', mininterval=2):
            # Iterate over the update steps for the current task
            sum_loss_q = 0 # Sum of losses for the current task
            # Define optimizer for task-level updates
            meta_optim_1 = optim.Adam(Maml_net.parameters(), lr=self.update_lr)

            # Train on support set
            for batch in x_spt:
                batch = batch.to(self.device)
                # Forward pass
                pred = Maml_net(batch)
                # Compute loss
                loss_s, prediction, ground_truth = loss_func(pred, batch)
                # Zero gradients
                meta_optim_1.zero_grad()
                # Backward pass and optimization
                loss_s.backward()
                meta_optim_1.step()
                # Clip gradients to prevent exploding gradients in RNNs or LSTMs
                torch.nn.utils.clip_grad_norm_(Maml_net.parameters(), 20)

            # Evaluate on query set
            for batch in x_qry:
                batch = batch.to(self.device)
                pred = Maml_net(batch)
                loss_q, prediction, ground_truth = loss_func(pred, batch)
                sum_loss_q += loss_q.item()
            losses_q[k] += sum_loss_q
        ave_loss_q = torch.tensor(losses_q[-1], requires_grad=True, device=self.device)

        # grads = torch.autograd.grad(ave_loss_q, Maml_net.parameters())

        # 计算损失对 MAML_net 参数的梯度
        ave_loss_q.backward()  # 在 MAML_net 上计算查询集的损失并计算梯度
        # for name, param in Maml_net.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is None:
        #             print(f"No gradient for {name}")
        #         else:
        #             print(f"Gradient exists for {name}")

        # 获取梯度
        maml_gradients = [param.grad.clone() for param in Maml_net.parameters()]

        # 手动更新 self.net 的参数
        with torch.no_grad():
            for param, grad in zip(self.net.parameters(), maml_gradients):
                updated_param = param - grad.to(param.device) * self.meta_lr
                param.data.copy_(updated_param)  # 将更新后的参数复制回原参数
                # print(param)

        # # 获取更新后的参数的状态字典
        # updated_state_dict = {name: param.data for name, param in self.net.named_parameters()}
        # # 将更新后的状态字典加载到 self.net 中
        # self.net.load_state_dict(updated_state_dict)


        return ave_loss_q, self.net
