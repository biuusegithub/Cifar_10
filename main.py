import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch.utils.data as data




'''整理数据和标签'''
def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]

    tokens = [l.rstrip().split(',') for l in lines]     # .rstrip()用于去除结尾字符
    return dict(((name, label) for name, label in tokens))



'''划分训练集和测试集'''
data_dir = "data\cifar-10"
# labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
# print(labels)


# 文件拷贝
def copyfile(filename, targetdir):
    os.makedirs(targetdir, exist_ok=True)
    shutil.copy(filename, targetdir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    # 算出数量最少的类别
    n = collections.Counter(labels.values()).most_common()[-1][1]

    # 验证集中每个类被样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))

    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]    # 取出实际标签 如 cat
        fname = os.path.join(data_dir, 'train', train_file)

        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))

        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))

    return n_valid_per_label


'''验证集'''
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        fname = os.path.join(data_dir, 'test', test_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


def reorg_cifar_10(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
    print("finish")


batch_size, valid_ratio = 256, 0.1
# reorg_cifar_10(data_dir, valid_ratio)



'''图像增广'''
train_transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.42, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])

test_transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])



'''读取数据集'''
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
    transform = train_transformer) for folder in ['train', 'train_valid']
]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
    transform = test_transformer) for folder in ['valid', 'test']
]


train_iter, train_valid_iter = [
    data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    ) for dataset in (train_ds, train_valid_ds)
]

valid_iter = data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

test_iter = data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)



'''定义模型'''
def get_net():
    pretrain_net = torchvision.models.resnet18(pretrained=True)
    pretrain_net.fc = nn.Linear(pretrain_net.fc.in_features, 10)
    nn.init.xavier_uniform_(pretrain_net.fc.weight)
    return pretrain_net

loss = nn.CrossEntropyLoss(reduction = 'none')



def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')



devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()

# train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)
# d2l.plt.show()


'''当用train_iter, valid_iter调试好模型超参数后, 即可用全部数据train_valid_iter进行训练'''
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
d2l.plt.show()



'''对测试集进行预测并提交'''
preds = []

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})       # 这里并不知道为什么要这样排序
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)



