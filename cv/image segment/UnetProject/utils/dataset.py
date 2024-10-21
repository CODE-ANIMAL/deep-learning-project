from torch.utils.data import Dataset, DataLoader
import os
import glob
import cv2
from matplotlib import pyplot as plt
    
class getdata(Dataset):
    def __init__(self, directory_path) -> None:
        super(getdata, self).__init__()
        self.images_path = directory_path+'Train_Images/'
        self.labels_path = directory_path+'Train_Labels/'
        self.images_name = [i for i in os.listdir(self.images_path) if i.endswith('.png') or i.endswith('.jpg')]
        self.labels_name = [i for i in os.listdir(self.labels_path) if i.endswith('.png') or i.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(self, index):
        # 根据路径读取图片文件
        image = cv2.imread(self.images_path+self.images_name[index])
        label = cv2.imread(self.labels_path+self.labels_name[index])
        
        # label从rgb图转为灰度图，为了方便等下转为二值图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
        # _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 对label进行归一化处理，使得值位于0和1之间
        if label.max() > 1:
            label = label / 255
        # 通道数为1，要把Chanel放到前面
        
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label
def visualize_images(image):
    """可视化原始图像和分割结果，并显示评估指标"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Image')
    axs[0].axis('off')
    plt.show()


if __name__ == '__main__':
    data_path = '/Users/liumeng/Desktop/files/code/Breast/MYMODEL/dataset/segment dataset/'
    dataset = getdata(data_path)
    train_loader = DataLoader(dataset=dataset,
                                batch_size=16, 
                                shuffle=True)
    for i, (image, label) in enumerate(train_loader):        
        # shape of image or label is [16, 1, 512, 512]
        print(image.shape)
        break
