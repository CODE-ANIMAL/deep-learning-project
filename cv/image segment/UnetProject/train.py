from utils.dataset import getdata
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# from model2.img_segment import UNet
from img_segment import UNet
from torch.utils.data import DataLoader

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

if __name__ == '__main__':
    # 加载数据
    data_path = '/Users/liumeng/Desktop/files/code/Breast/MYMODEL/dataset/segment dataset/'
    data = getdata(data_path)
    batch_size = 16
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1).to(device)  # 输入rgb，输出1分类
    epochs = 10
    each_epoch_size = len(data) / batch_size
    lr = 0.001
    # 定义RMSprop算法
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    with tqdm(total=epochs*each_epoch_size) as pbar:
        for epoch in range(epochs):
            # 训练模式
            model.train()
            # 按照batch_size开始训练
            for i, (image, label) in enumerate(data_loader):
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = model(image)
                # 计算loss
                loss = criterion(pred, label)
                # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), 'model2/SEGMENT_model.pth')
                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)
