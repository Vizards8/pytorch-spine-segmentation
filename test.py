from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

# file = './dataset/train/source/Case1.nii.gz'  # 你的nii或者nii.gz文件路径
# labels =[0 for i in range(20)]
# for i in range(1,201):
#     file = './dataset/train/label/mask_case'+str(i)+'.nii.gz'  # 你的nii或者nii.gz文件路径
#     img = nib.load(file)
#     data = img.get_data()
#
#     import numpy as np
#
#     uni = np.unique(data)
#     # print(uni)
#     for j in uni:
#         labels[j] += 1
# print('Calculate: number of MRI with this label / total number of MRI')
# for i in range(20):
#     print(f'{i}: {labels[i]}/200')

# print('img:', img)
# print('header:', img.header['db_name'])  # 输出nii的头文件

# width, height, queue = img.dataobj.shape

# OrthoSlicer3D(img.dataobj).show()

# num = 1
# for i in range(queue):
#     img_arr = img.dataobj[:, :, i]
#     plt.subplot(5, 3, i + 1)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1
#
# plt.show()

# import SimpleITK as sitk
#
# file = '../dataset/train/source/Case1.nii.gz'  # 你的nii或者nii.gz文件路径
# # file = '../dataset/train/label/mask_case1.nii.gz'  # 你的nii或者nii.gz文件路径
# labelImage = sitk.ReadImage(file)  # in_file是nii.gz文件的路径
# imag_results = sitk.GetArrayFromImage(labelImage)
# spacing = labelImage.GetSpacing()
# origin = labelImage.GetOrigin()
# print('spacing is:', spacing)  # spacing和orgin是写入的内容
# print('origin is:', origin)
#
# # file = '../dataset/train/source/Case1.nii.gz'  # 你的nii或者nii.gz文件路径
# file = '../dataset/train/label/mask_case1.nii.gz'  # 你的nii或者nii.gz文件路径
# labelImage = sitk.ReadImage(file)  # in_file是nii.gz文件的路径
# imag_results = sitk.GetArrayFromImage(labelImage)
# spacing = labelImage.GetSpacing()
# origin = labelImage.GetOrigin()
# print('spacing is:', spacing)  # spacing和orgin是写入的内容
# print('origin is:', origin)

# from GetData import GetLoader
# from torch.utils.data import DataLoader
#
# dataset = GetLoader('../dataset/train/source/', '../dataset/train/label/')
# train_loader = DataLoader(dataset,
#                           batch_size=12,
#                           shuffle=True,
#                           pin_memory=True,
#                           drop_last=False)
# for i, batch in enumerate(train_loader):
#     print(i)
#     # print(batch['image'].shape)

# import torch
# import torch.nn as nn
#
# model = nn.Linear(10, 3)
# criterion = nn.CrossEntropyLoss()
#
# x = torch.randn(16, 10)
# y = torch.randint(0, 3, size=(16,))  # (16, )
# logits = model(x)  # (16, 3)
#
# loss = criterion(logits, y)
# print(loss)

# import torchio as tio
#
# sampler = tio.GridSampler(patch_size=(256, 256, 1))
# # colin = tio.datasets.Colin27()
# colin = tio.Subject(
#     source=tio.ScalarImage('./dataset/slice_train/source/Case1_7.nii.gz'),
#     label=tio.LabelMap('./dataset/slice_train/label/mask_case1_7.nii.gz'),
# )
# for i, patch in enumerate(sampler(colin)):
#     patch.source.save(f'patch_{i}.nii.gz')
# # To figure out the number of patches beforehand:
# sampler = tio.GridSampler(subject=colin, patch_size=(256, 256, 1))
# print(len(sampler))

# # 测试两个张图的dice
# from utils.metrics import *
# import nibabel as nib
# import numpy as np
# from onehot import *
# import torch
#
# file = './dataset/dice/predict.nii.gz'
# img = nib.load(file)
# pred = img.get_fdata()
# file = './dataset/dice/gt.nii.gz'
# img = nib.load(file)
# gt = img.get_fdata()
#
# class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# pred = pred[np.newaxis, np.newaxis, :]
# pred = mask2onehot(pred, class_list)
# pred = torch.FloatTensor(pred).to('cuda:0')
# gt = gt[np.newaxis, np.newaxis, :]
# gt = mask2onehot(gt, class_list)
# gt = torch.FloatTensor(gt).to('cuda:0')
# dice_list = []
# for i in range(len(class_list)):
#     if i == 8:
#         print('a')
#     dice = diceCoeffv2(pred[:, i, :, :], gt[:, i, :, :])
#     dice_list.append(dice)
#     print(f'id:{i}, dice:{dice}')
# print(mean(dice_list))
