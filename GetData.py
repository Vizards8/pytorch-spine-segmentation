import os
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


class GetLoader(Dataset):
    # 初始化函数，得到数据
    def __init__(self, source_dir, label_dir):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.filename = os.listdir(self.source_dir)  # 文件名
        self.preprocess = transforms.Compose([transforms.RandomResizedCrop((880, 880)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])

        # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回

    def __getitem__(self, index):
        path = self.source_dir + self.filename[index]

        # import nibabel as nib
        # img = nib.load(path)
        # data_array = self.preprocess(img.dataobj[:, :, 5])

        data = nib.load(path)
        # print('image:', sitk.GetArrayFromImage(data).shape)
        data_array = sitk.GetArrayFromImage(data)[5, :, :]
        data_array_resize = np.resize(np.array(data_array, dtype=np.float64), (1, 256, 256))

        path = self.label_dir + 'mask_' + self.source[index]
        label = sitk.ReadImage(path)
        # print('mask:', sitk.GetArrayFromImage(label).shape)
        label_array = sitk.GetArrayFromImage(label)[5, :, :]
        label_array_resize = np.resize(np.array(label_array, dtype=np.float64), (1, 256, 256))

        return {
            'image': torch.from_numpy(data_array_resize).type(torch.FloatTensor),
            'mask': torch.from_numpy(label_array_resize).type(torch.FloatTensor)
        }
        # return data_array_resize, label_array_resize

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.filename)


# array[array_condition] = value
# an_array[an_array % 2 == 0] = 0
from scipy.ndimage import zoom

