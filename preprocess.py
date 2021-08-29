import nibabel as nib
from hparam import hparams as hp
import os
import numpy as np


# Load a nii.gz file from old_path, then slice it and save new files to the new_path
# For example:
# old_path = './Case1.nii.gz' will be saved as 'new_path + Case1_1.nii.gz' after sliced

def Slice(old_path, new_path):
    os.makedirs(new_path, exist_ok=True)
    name = old_path.split('/')[-1].split('.')[0]
    nii_img = nib.load(old_path)
    nii_data = nii_img.get_fdata()

    # 把仿射矩阵和头文件都存下来
    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()

    # 省略一些处理data的骚操作,比如：
    # new_data[new_data>0] = 1

    width, height, queue = nii_img.dataobj.shape
    for i in range(queue):
        # unique_data = np.unique(nii_data[:, :, i])
        # if unique_data.max() <= 17:
        print((np.unique(nii_data[:, :, i])).max())
        # continue
        new_data = nii_data[:, :, i].copy()

        # 形成新的nii文件
        new_nii = nib.Nifti1Image(new_data, affine, hdr)

        # 保存nii文件，后面的参数是保存的文件名
        new_name = name + '_' + str(i + 1) + hp.save_arch
        nib.save(new_nii, os.path.join(new_path, new_name))
        print('Saved to ' + new_path + new_name)


if __name__ == '__main__':
    source_train_dir = './dataset/train/source/'
    label_train_dir = './dataset/train/label/'
    source_test_dir = './dataset/test/source/'
    label_test_dir = './dataset/test/label/'

    files = os.listdir(source_train_dir)
    for file in files:
        Slice(source_train_dir + file, hp.source_train_dir)
    files = os.listdir(label_train_dir)
    for file in files:
        Slice(label_train_dir + file, hp.label_train_dir)

    files = os.listdir(source_test_dir)
    for file in files:
        Slice(source_test_dir + file, hp.source_test_dir)
    files = os.listdir(label_test_dir)
    for file in files:
        Slice(label_test_dir + file, hp.label_test_dir)

    # # For the test
    # old_path = '../dataset/train/source/Case200.nii.gz'
    # new_path = '../dataset/testslice/source/'
    # Slice(old_path, new_path)
