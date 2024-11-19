zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
# 定义字典，39 个元素
# {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, 'q': 13, 'w': 14, 'e': 15, 'r': 16, 't': 17, 'y': 18, 'u': 19, 'i': 20, 'o': 21, 'p': 22, 'a': 23, 's': 24, 'd': 25, 'f': 26, 'g': 27, 'h': 28, 'j': 29, 'k': 30, 'l': 31, 'z': 32, 'x': 33, 'c': 34, 'v': 35, 'b': 36, 'n': 37, 'm': 38} 
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}

# 定义列表，把所有元素存在列表里
# ['<SOS>', '<EOS>', '<PAD>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'] 
zidian_xr = [k for k, v in zidian_x.items()]

# 定义字典，39个元素的大写
#  {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, 'Q': 13, 'W': 14, 'E': 15, 'R': 16, 'T': 17, 'Y': 18, 'U': 19, 'I': 20, 'O': 21, 'P': 22, 'A': 23, 'S': 24, 'D': 25, 'F': 26, 'G': 27, 'H': 28, 'J': 29, 'K': 30, 'L': 31, 'Z': 32, 'X': 33, 'C': 34, 'V': 35, 'B': 36, 'N': 37, 'M': 38} 
zidian_y = {k.upper(): v for k, v in zidian_x.items()}
# 定义列表，39个元素的大写存在列表中
#  ['<SOS>', '<EOS>', '<PAD>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M']
zidian_yr = [k for k, v in zidian_y.items()]

import random
import numpy as np
import torch


def get_data():
    # 定义词集合
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]

    # 定义每个词被选中的概率
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    # 随机选n个词，范围在30～48之间
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p)

    # 采样的结果就是x
    x = x.tolist()

    # y是对x的变换得到的
    # 字母大写,数字取10以内的互补数
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)

    y = [f(i) for i in x]
    y = y + [y[-1]]  # 最后一位重复两位
    # 逆序
    y = y[::-1]

    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # 编码成数据，本来是字母组成的列表，现在处理成字母对应着的编号
    x = [zidian_x[i] for i in x]
    y = [zidian_y[i] for i in y]

    # 转tensor，将数字类型的列表转换成tensor类型的数据
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


# 两数相加测试,使用这份数据请把main.py中的训练次数改为10
# def get_data():
#     # 定义词集合
#     words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#     # 定义每个词被选中的概率
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     p = p / p.sum()
#
#     # 随机选n个词
#     n = random.randint(10, 20)
#     s1 = np.random.choice(words, size=n, replace=True, p=p)
#
#     # 采样的结果就是s1
#     s1 = s1.tolist()
#
#     # 同样的方法,再采出s2
#     n = random.randint(10, 20)
#     s2 = np.random.choice(words, size=n, replace=True, p=p)
#     s2 = s2.tolist()
#
#     # y等于s1和s2数值上的相加
#     y = int(''.join(s1)) + int(''.join(s2))
#     y = list(str(y))
#
#     # x等于s1和s2字符上的相加
#     x = s1 + ['a'] + s2
#
#     # 加上首尾符号
#     x = ['<SOS>'] + x + ['<EOS>']
#     y = ['<SOS>'] + y + ['<EOS>']
#
#     # 补pad到固定长度
#     x = x + ['<PAD>'] * 50
#     y = y + ['<PAD>'] * 51
#     x = x[:50]
#     y = y[:51]
#
#     # 编码成数据
#     x = [zidian_x[i] for i in x]
#     y = [zidian_y[i] for i in y]
#
#     # 转tensor
#     x = torch.LongTensor(x)
#     y = torch.LongTensor(y)
#
#     return x, y


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 16000

    def __getitem__(self, i):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset(),
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)

