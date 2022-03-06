#使用vgg16预训练模型 微调自己的数据集
"""
1、加载自己的数据
2、加载预训练模型并修改分类器网络层
3、保存、调用训练好的模型和参数

""" 


import torch
from torch import nn,optim,tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets,transforms
import numpy as np
from matplotlib import pyplot as plt
import time
from torchvision import models


#全局变量
batch_size = 32 # 每次喂入的数据量
# num_print = int(50000//batch_size//4)  #每n次batch打印一次
num_print = 2
epoch_num = 100  #总迭代次数
lr = 0.01        #学习率
step_size = 10  #每n个epoch更新一次学习率

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

#定义自己数据的文件夹 
train_dir = './dataset'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

# 要是有验证集的话 数据集结构如下
# - data 
# -- train
# --- cat
# --- dog
# -- val
# --- cat
# --- dog

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 用预训练模型 改变最后分类层的结构
class VGGNet(nn.Module):
    def __init__(self, num_classes=8):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        middle = x
        x = self.classifier(x)
        return x,middle # 输出包括 最后的向量

#--------------------训练过程---------------------------------
model = VGGNet().to(device)

if torch.cuda.is_available():
    model.cuda()
# 可训练的参数为分类器的参数
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = lr,momentum = 0.8,weight_decay = 0.001 )
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

loss_list = []
start = time.time()

# train
for epoch in range(epoch_num):  
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs ,labels = inputs.to(device),labels.to(device)
        
        optimizer.zero_grad()   
        outputs,middle = model(inputs)
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loss_list.append(loss.item())
        if i % num_print == num_print-1 :
            print('[%d epoch, %d] loss: %.6f' %(epoch + 1, i + 1, running_loss / num_print))
            running_loss = 0.0  
    lr_1 = optimizer.param_groups[0]['lr']
    print('learn_rate : %.15f'%lr_1)
    scheduler.step()

end = time.time()
# print('time:{}'.format(end-start))

torch.save(model, './model.pkl')   #保存模型
#model = torch.load('./model.pkl')  #加载模型

# # loss images show
# plt.plot(loss_list, label='Minibatch cost')
# plt.plot(np.convolve(loss_list,np.ones(200,)/200, mode='valid'),label='Running average')
# plt.ylabel('Cross Entropy')
# plt.xlabel('Iteration')
# plt.legend()
# plt.show()

# test
model.eval()
correct = 0.0
total = 0
with torch.no_grad():  # 训练集不需要反向传播
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
        outputs,middle = model(inputs)
        pred = outputs.argmax(dim = 1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)
        correct += torch.eq(pred,labels).sum().item()
print('Accuracy of the network on test images: %.2f %%' % (100.0 * correct / total))
# save tensor
torch.save(middle, "./myTensor.pt")
# load tensor
# y = torch.load("./myTensor.pt")
# print(y)
