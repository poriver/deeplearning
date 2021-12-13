import torch
from torch import nn
from my_dataset2 import get_dataloader
import lib
import os
import numpy as np
from torch.nn import functional as F
from my_resnet2 import RestNet18
from my_resnet import resnet_test


model = resnet_test()
# print(model)
model.to(lib.device)

# x = torch.ones((3, 224, 224), dtype=torch.float32)
# x = torch.reshape(x, (-1, 3, 224, 224))
# print(x.shape)
# output = model(x)
# print(output.shape)

# loss_fn = F.nll_loss()
# loss_fn.to(lib.device)

optimizer = torch.optim.Adam(model.parameters(), lr=lib.learning_rate)

if os.path.exists("./my_model.pkl"):
    model.load_state_dict(torch.load("./my_model.pkl"))
    optimizer.load_state_dict(torch.load("./optimizer.pkl"))

def train(epoch):
    print("使用{}训练".format(lib.device))
    print("-------第{}轮训练开始-------".format(i + 1))

    # 训练步骤开始
    model.train()
    for idx, (input, target) in enumerate(get_dataloader(train=True)):
        # 放入gpu训练
        input = input.to(lib.device)
        target = target.to(lib.device)
        # 梯度归零
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print(epoch, idx, loss.item())

        if idx % 100 == 0:
            # print("训练次数:{}，Loss:{}".format(idx, loss.item()))
            torch.save(model.state_dict(), "./my_model.pkl")
            torch.save(optimizer.state_dict(), "./optimizer.pkl")

def eval():
    model.eval()
    loss_list = []
    acc_list = []
    for idx, (input, target) in enumerate(get_dataloader(train=False)):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.cpu().item())
            # 计算准确率
            pred = output.max(dim=-1)[-1]
            acc = pred.eq(target).float().mean()
            print(acc.item())
            acc_list.append(acc.cpu().item())

    print("total loss,acc", np.mean(loss_list), np.mean(acc_list))
    # torch.save(model.state_dict(), "./my_model.pkl")
    # torch.save(optimizer.state_dict(), "./optimizer.pkl")
    # print("模型已保存")

if __name__ == '__main__':
    # for i in range(1):
    #     train(i)
    eval()







