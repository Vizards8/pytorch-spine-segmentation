from glob import glob
from os.path import dirname, join, basename, isfile
import sys

sys.path.append('./')
import csv
import torch
# from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler, WeightedSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from hparam import hparams as hp
from tqdm import tqdm


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob(hp.fold_arch))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

        val_size = int(hp.val_split * len(self.image_paths))
        test_size = int(hp.test_split * len(self.image_paths))
        train_size = len(self.image_paths) - val_size - test_size

        self.subjects_paths = []  # dict打包路径，用于split
        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject_path = {
                'source': image_path,
                'label': label_path,
            }
            self.subjects_paths.append(subject_path)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.subjects_paths,
            [train_size, val_size, test_size],
            torch.Generator().manual_seed(0))

        # split完成，train,val分别从各自dataset中的路径加载
        print('Loading Dataset......\n')

        self.train_dataset = self.load_from_dataset(train_dataset)
        self.val_dataset = self.load_from_dataset(val_dataset)

        # # 一次性加载所有data，内存开销巨大，口区
        # self.subjects = []
        #
        # for (image_path, label_path) in zip(self.image_paths, self.label_paths):
        #     subject = tio.Subject(
        #         source=tio.ScalarImage(image_path),
        #         label=tio.LabelMap(label_path),
        #     )
        #     torch.round(subject['label']['data'], out=subject['label']['data'])  # 处理为int
        #
        #     if not np.equal(subject['source']['affine'], subject['label']['affine']).all:
        #         print("ERROR Handling:Affine not equal!", image_path.split('/')[-1])
        #     else:
        #         self.subjects.append(subject)
        #
        # self.transforms = self.transform()
        #
        # self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        #
        # self.queue_dataset = Queue(
        #     self.training_set,
        #     queue_length,
        #     samples_per_volume,
        #     UniformSampler(patch_size),
        # )

    def load_from_dataset(self, dataset):
        # dataset 为list中嵌套字典，字典包含两个路径'source'与'label'

        if hp.mode == '3d':
            patch_size = hp.patch_size
        elif hp.mode == '2d':
            # patch_size = (hp.patch_size,hp.patch_size,1)
            patch_size = hp.patch_size
        else:
            raise Exception('no such kind of mode!')

        subjects = []

        queue_length = 5
        samples_per_volume = hp.samples_per_volume

        for i in tqdm(dataset):
            subject = tio.Subject(
                source=tio.ScalarImage(i['source']),
                label=tio.LabelMap(i['label']),
            )
            torch.round(subject['label']['data'], out=subject['label']['data'])  # 处理为int

            if not np.equal(subject['source']['affine'], subject['label']['affine']).all:
                print("ERROR Handling:Affine not equal!", image_path.split('/')[-1])
            else:
                subjects.append(subject)

        subjects_set = tio.SubjectsDataset(subjects, transform=self.transform())

        if hp.use_queue:
            queue_dataset = Queue(
                subjects_set,
                queue_length,
                samples_per_volume,
                UniformSampler(patch_size),
            )
            return queue_dataset
        else:
            return subjects_set

    # def __init__(self, images_dir, labels_dir):
    #
    #     if hp.mode == '3d':
    #         patch_size = hp.patch_size
    #     elif hp.mode == '2d':
    #         patch_size = hp.patch_size
    #     else:
    #         raise Exception('no such kind of mode!')
    #
    #     images_dir = Path(images_dir)
    #     image_paths = sorted(images_dir.glob(hp.fold_arch))
    #     labels_dir = Path(labels_dir)
    #     label_paths = sorted(labels_dir.glob(hp.fold_arch))
    #     self.training_set = []
    #     for (image_path, label_path) in zip(image_paths, label_paths):
    #         self.training_set.append({'source': image_path, 'label': label_path})
    #
    # def __getitem__(self, index):
    #
    #     subject = tio.Subject(
    #         source=tio.ScalarImage(self.dataset[index]['source']),
    #         label=tio.LabelMap(self.dataset[index]['label']),
    #     )
    #     torch.round(subject['label']['data'], out=subject['label']['data'])  # 处理为int
    #
    #     if not np.equal(subject['source']['affine'], subject['label']['affine']).all:
    #         print("ERROR Handling:Affine not equal!", image_path.split('/')[-1])
    #
    #     self.transforms = self.transform()
    #
    #     self.training_set = tio.SubjectsDataset(subject, transform=self.transforms)
    #
    #     return self.training_set

    def transform(self):

        if hp.mode == '3d':
            if hp.aug:
                training_transform = Compose([
                    # ToCanonical(),
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomMotion(),
                    RandomBiasField(),
                    ZNormalization(),
                    RandomNoise(),
                    RandomFlip(axes=(0,)),
                    OneOf({
                        RandomAffine(): 0.8,
                        RandomElasticDeformation(): 0.2,
                    }), ])
            else:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    ZNormalization(),
                ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomMotion(),
                    RandomBiasField(),
                    ZNormalization(),
                    RandomNoise(),
                    RandomFlip(axes=(0,)),
                    # OneOf({
                    #     RandomAffine(): 0.8,
                    #     RandomElasticDeformation(): 0.2,
                    # }),
                ])
            else:
                training_transform = Compose([
                    CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')

        return training_transform


class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        self.subjects = []

        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob(hp.fold_arch))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)

        # self.transforms = self.transform()

        self.test_dataset = tio.SubjectsDataset(self.subjects, transform=None)
