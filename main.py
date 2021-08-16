import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metrics import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import onehot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_dir_test = hp.output_dir_test


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    return parser


def train():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        from models.two_d.unet import Unet
        model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.two_d.miniseg import MiniSeg
        # model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        # from models.two_d.deeplab import DeepLabV3
        # model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':

        from models.three_d.unet3d import UNet3D
        model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)

        # from models.three_d.residual_unet3d import UNet
        # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2)

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from utils.loss import SoftDiceLoss
    criterion = SoftDiceLoss(hp.out_class).cuda()

    writer = SummaryWriter(args.output_dir)

    # from GetData import GetLoader
    # train_dataset = GetLoader(source_train_dir, label_train_dir)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch,
    #                           shuffle=True,
    #                           pin_memory=False,
    #                           drop_last=False)

    full_dataset = MedData_train(source_train_dir, label_train_dir)
    val_size = int(hp.val_split * len(full_dataset.training_set))
    test_size = int(hp.test_split * len(full_dataset.training_set))
    train_size = len(full_dataset.training_set) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset.training_set,
                                                                             [train_size, val_size, test_size],
                                                                             torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False)
    print('Train:', len(train_dataset))
    print('val:', len(val_dataset))
    print('Data Loaded! Prepare to train......')

    # train_loader = DataLoader(train_dataset.queue_dataset,
    #                           batch_size=args.batch,
    #                           shuffle=True,
    #                           pin_memory=False,
    #                           drop_last=False)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    val_iteration = elapsed_epochs * len(val_loader)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        end_time = time.time()
        print("epoch:" + str(epoch) + "duration:" + str(end_time - start_time))
        start_time = end_time

        epoch += elapsed_epochs

        num_iters = 0

        for i, batch in enumerate(train_loader):

            if hp.debug:
                if i >= 1:
                    break

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()

            x = batch['source']['data']
            y = batch['label']['data']

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

                # y[y != 0] = 1
            print(y.max())

            print('before turn 2 onehot:', np.unique(np.array(y)))
            y = onehot.mask2onehot(y, hp.out_classlist)
            x = torch.FloatTensor(x).cuda()
            y = torch.FloatTensor(y).cuda()

            outputs = model(x)
            outputs = torch.sigmoid(outputs)

            # for metrics
            # logits = torch.sigmoid(outputs)
            # labels = logits.clone()
            # labels[labels > 0.5] = 1
            # labels[labels <= 0.5] = 0
            loss = criterion(outputs, y)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1

            # class_false_positive_rate = []
            # class_false_negtive_rate = []
            # labels = outputs.clone()
            # labels = onehot.onehot2mask(labels.cpu().detach().numpy())
            # labels = onehot.mask2onehot(labels, hp.out_classlist)
            # labels = torch.FloatTensor(labels).cuda()
            # for j in range(hp.out_class):
            #     false_positive_rate, false_negtive_rate = FP_FN_metric(labels.cpu()[:, i:i + 1, :, :],
            #                                                            y.cpu()[:, i:i + 1, :, :])
            #     class_false_positive_rate.append(false_positive_rate)
            #     class_false_negtive_rate.append(false_negtive_rate)
            # print('class_false_negtive_rate:', class_false_positive_rate)
            # print('class_false_negtive_rate:', class_false_negtive_rate)
            # mean_class_false_positive_rate = sum(class_false_positive_rate) / len(class_false_positive_rate)
            # mean_class_false_negtive_rate = sum(class_false_negtive_rate) / len(class_false_negtive_rate)

            ## log
            writer.add_scalar('Training/Loss', loss.item(), iteration)
            # writer.add_scalar('Training/false_positive_rate', mean_class_false_positive_rate, iteration)
            # writer.add_scalar('Training/false_negtive_rate', mean_class_false_negtive_rate, iteration)
            # writer.add_scalar('Training/dice', dice, iteration)

            print("loss:" + str(loss.item()))
            print('lr:' + str(scheduler._last_lr[0]))

        scheduler.step()

        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {

                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )

            with torch.no_grad():

                # x [BS,1,880,880]
                # y [BS,18,880,880]
                if hp.mode == '2d':
                    x = x.unsqueeze(4)
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)

                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                y = y[np.newaxis, :, :, :, :]
                outputs = outputs[0].cpu().detach().numpy()
                outputs = outputs[np.newaxis, :, :, :, :]
                affine = batch['source']['affine'][0].numpy()

                y = onehot.onehot2mask(y)[0]
                print('turn back from onehot:', np.unique(y))
                outputs = onehot.onehot2mask(outputs)[0]

                # x [1,880,880,1]
                # y [1,880,880,1]
                # outputs [1,880,880,1]
                source_image = torchio.ScalarImage(tensor=x, affine=affine)
                source_image.save(os.path.join(args.output_dir, f"step-{epoch:04d}-source" + hp.save_arch))
                # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                label_image = torchio.ScalarImage(tensor=y, affine=affine)
                label_image.save(os.path.join(args.output_dir, f"step-{epoch:04d}-gt" + hp.save_arch))

                output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                output_image.save(os.path.join(args.output_dir, f"step-{epoch:04d}-predict" + hp.save_arch))

        # Validation per epoch
        with torch.no_grad():
            print('Validation............')
            total_loss = []
            for i, batch in enumerate(val_loader):
                # print(f"Batch: {i}/{len(val_loader)} epoch {epoch}")

                x = batch['source']['data']
                y = batch['label']['data']

                if hp.mode == '2d':
                    x = x.squeeze(4)
                    y = y.squeeze(4)
                # x [BS,1,880,880]
                # y [BS,1,880,880]

                # print('before turn 2 onehot:', np.unique(np.array(y)))
                y = onehot.mask2onehot(y, hp.out_classlist)
                x = x.type(torch.FloatTensor).cuda()
                y = torch.FloatTensor(y).cuda()
                # y [BS,18,880,880]

                outputs = model(x)
                outputs = torch.sigmoid(outputs)
                val_loss = criterion(outputs, y)
                val_iteration += 1
                writer.add_scalar('Training/Val_Loss', val_loss.item(), val_iteration)

                # print("val_loss:" + str(val_loss.item()))
                total_loss.append(val_loss.item())

            print("mean_val_loss:" + str(np.mean(np.array(total_loss))))

            if epoch % args.epochs_per_checkpoint == 0:
                # x [BS,1,880,880]
                # y [BS,18,880,880]
                # outputs [BS,18,880,880]
                if hp.mode == '2d':
                    x = x.unsqueeze(4)
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)

                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                y = y[np.newaxis, :, :, :, :]
                outputs = outputs[0].cpu().detach().numpy()
                outputs = outputs[np.newaxis, :, :, :, :]
                affine = batch['source']['affine'][0].numpy()

                y = onehot.onehot2mask(y)[0]
                print('turn back from onehot:', np.unique(y))
                outputs = onehot.onehot2mask(outputs)[0]

                # x [1,880,880,1]
                # y [1,880,880,1]
                # outputs [1,880,880,1]
                source_image = torchio.ScalarImage(tensor=x, affine=affine)
                source_image.save(os.path.join(args.output_dir, f"val-step-{epoch:04d}-source" + hp.save_arch))
                # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                label_image = torchio.ScalarImage(tensor=y, affine=affine)
                label_image.save(os.path.join(args.output_dir, f"val-step-{epoch:04d}-gt" + hp.save_arch))

                output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                output_image.save(os.path.join(args.output_dir, f"val-step-{epoch:04d}-predict" + hp.save_arch))

    writer.close()


def test():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_dir_test, exist_ok=True)

    if hp.mode == '2d':
        from models.two_d.unet import Unet
        model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.two_d.miniseg import MiniSeg
        # model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        # from models.two_d.deeplab import DeepLabV3
        # model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':
        from models.three_d.unet3d import UNet
        model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2)

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

    model = torch.nn.DataParallel(model, device_ids=devicess, output_device=[1])

    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    model.cuda()

    test_dataset = MedData_test(source_test_dir, label_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size

    for i, subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
            subj,
            patch_size,
            patch_overlap,
        )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=16)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):

                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)
                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels > 0.5] = 1
                labels[labels <= 0.5] = 0

                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()

        affine = subj['source']['affine']

        label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
        label_image.save(os.path.join(output_dir_test, f"{str(i)}-result_float" + hp.save_arch))

        # f"{str(i):04d}-result_float.mhd"

        output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
        output_image.save(os.path.join(output_dir_test, f"{str(i)}-result_int" + hp.save_arch))


if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
