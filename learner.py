# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, device, dropout):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.dropout = dropout

        self.LSTM = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.layer_size,
                            batch_first=True,
                            # dropout=self.dropout,
                            bidirectional=False)
        # 定义 Dropout 层
        self.dropout_layer = nn.Dropout(dropout)

        self.fc = nn.Linear(self.hidden_size,
                            self.output_size)


        self.sig = nn.Sigmoid()
        self.device = device
            # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化LSTM权重
        for name, param in self.LSTM.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param.data)
            elif 'bias' in name:
                init.constant_(param.data, 0.0)

        # 初始化全连接层权重
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0.0)



    # input [batch_size,length,question*2]
    def forward(self, inputs):
        # output [batch_size,length,hinddens]
        output, _ = self.LSTM(inputs)
        output = self.dropout_layer(output)
        output = self.fc(output)
        logits = self.sig(output)
        # logits = torch.relu(output)
        # input[batch_size, length, question]
        return logits