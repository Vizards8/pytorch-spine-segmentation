import os
from hparam import hparams as hp

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devices = [i for i in range(hp.gpu_nums)]

import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from utils.metrics import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import onehot
from data_function import MedData_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(f'small_sample:{hp.small_sample}, model:{hp.model_name}, out_class:{hp.out_class}')

    device_name = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available():
        print(f'Using Device: {torch.cuda.device_count()} GPU {device_name}')
    else:
        print(f'Using Device:{device}')

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        if hp.model_name == 'SegNet':
            from models.two_d.segnet2 import SegNet
            model = SegNet(n_init_features=hp.in_class, num_classes=hp.out_class)

        elif hp.model_name == 'UNet':
            from models.two_d.unet import Unet
            model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        elif hp.model_name == 'MiniSeg':
            from models.two_d.miniseg import MiniSeg
            model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        elif hp.model_name == 'PSPNet':
            from models.two_d.pspnet import PSPNet
            model = PSPNet(in_class=hp.in_class, n_classes=hp.out_class)

        elif hp.model_name == 'AttUNet':
            # ATTU_Net 512*512*1
            from models.two_d.attunet import AttU_Net
            model = AttU_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'R2UNet':
            # R2U_Net 512*512*1
            from models.two_d.R2U_Net import R2U_Net
            model = R2U_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'R2AttUNet':
            # R2AttU_Net 265*256*1
            from models.two_d.R2AttU_Net import R2AttU_Net
            model = R2AttU_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'DeepLabv3':
            from models.two_d.deeplab2 import DeepLabv3_plus
            model = DeepLabv3_plus(nInputChannels=hp.in_class, n_classes=hp.out_class)

        elif hp.model_name == 'UNetpp':
            from models.two_d.unetpp import ResNet34UnetPlus
            model = ResNet34UnetPlus(num_channels=hp.in_class, num_class=hp.out_class)

        else:
            print('ERROR: No such model')
        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet  # 报错
        # model = SegNet(input_nbr=hp.in_class, label_nbr=hp.out_class)

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

    model = torch.nn.DataParallel(model, device_ids=devices)
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
                    state[k] = v.to(device)

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.to(device)

    from utils.loss import SoftDiceLoss
    criterion = SoftDiceLoss(hp.out_class).to(device)

    writer = SummaryWriter(args.output_dir)

    # from GetData import GetLoader
    # train_dataset = GetLoader(source_train_dir, label_train_dir)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch,
    #                           shuffle=True,
    #                           pin_memory=False,
    #                           drop_last=False)

    print('Loading Dataset......\n')
    full_dataset = MedData_train(hp.source_train_dir, hp.label_train_dir)
    train_loader = DataLoader(full_dataset.dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=hp.num_workers)

    full_dataset = MedData_train(hp.source_test_dir, hp.label_test_dir)
    val_loader = DataLoader(full_dataset.dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=hp.num_workers)
    print('\nTrainSet Total Number:', len(train_loader) * hp.batch_size)
    print('ValSet Total Number:', len(val_loader) * hp.batch_size)
    print('Data Loaded! Prepare to train......\n')

    # train_loader = DataLoader(train_dataset.queue_dataset,
    #                           batch_size=args.batch,
    #                           shuffle=True,
    #                           pin_memory=False,
    #                           drop_last=False)

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    val_iteration = elapsed_epochs * len(val_loader)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch += elapsed_epochs

        total_train_loss = []
        total_train_IOU = []

        loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in loop_train:

            model.train()

            optimizer.zero_grad()

            x = batch['source']['data']
            y = batch['label']['data']

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

            # print('before turn 2 onehot:', np.unique(np.array(y)))
            y = onehot.mask2onehot(y, hp.out_classlist)  # 转成one-hot
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)

            # # 查看模型详情，除非调试否则注释，占GPU内存的
            # from torchsummary import summary
            # print(summary(model, (1, 256, 256)))

            # Loss
            outputs = model(x)
            # outputs = torch.sigmoid(outputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # ss = []
            # for j in range(hp.out_class):
            #     ss.append(round(outputs[0, j, 0, 0].item(), 3))
            # print(ss)
            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()
            iteration += 1

            # for metrics
            predict = outputs.clone()
            predict = onehot.onehot2mask(predict.cpu().detach().numpy())
            if hp.see_predict:
                print(np.unique(predict))
            predict = onehot.mask2onehot(predict, hp.out_classlist)
            predict = torch.FloatTensor(predict).to(device)  # 转换为torch.tensor才能送进gpu
            IOU, dice, acc, false_positive_rate, false_negative_rate = metrics(predict, y, hp.out_class)

            # Log
            writer.add_scalar('Training/Loss', loss.item(), iteration)
            writer.add_scalar('Training/IOU', IOU.item(), iteration)
            writer.add_scalar('Training/Dice', dice.item(), iteration)
            writer.add_scalar('Training/Recall', acc.item(), iteration)
            writer.add_scalar('Training/False_Positive_rate', false_positive_rate.item(), iteration)
            writer.add_scalar('Training/False_Negative_rate', false_negative_rate.item(), iteration)

            # Set tqdm
            end_time = time.time()
            total_train_loss.append(loss.item())
            total_train_IOU.append(IOU.item())
            loop_train.set_description(f'Train [{epoch}/{epochs}]')
            loop_train.set_postfix({
                'loss': '{0:1.5f}'.format(loss.item()),
                'acc': '{0:1.5f}'.format(acc.item()),
                'duration': '{0:1.5f}'.format(end_time - start_time)
            })
            # print("loss:" + str(loss.item()))
            # print('lr:' + str(scheduler._last_lr[0]))

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

        # Save checkpoint and predicted *.nii.gz
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
                # print('turn back from onehot:', np.unique(y))
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
        model.eval()
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader))
            total_valid_loss = []
            total_valid_acc = []
            total_valid_IOU = []

            for i, batch in loop_val:
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
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor(y).to(device)
                # y [BS,18,880,880]

                # Loss
                outputs = model(x)
                outputs = torch.sigmoid(outputs)
                val_loss = criterion(outputs, y)
                val_iteration += 1

                # for metrics
                predict = outputs.clone()
                predict = onehot.onehot2mask(predict.cpu().detach().numpy())
                # print(np.unique(predict))
                predict = onehot.mask2onehot(predict, hp.out_classlist)
                predict = torch.FloatTensor(predict).to(device)  # 转换为torch.tensor才能送进gpu
                IOU, dice, acc, false_positive_rate, false_negative_rate = metrics(predict, y, hp.out_class)

                # Log
                writer.add_scalar('Validation/Val_Loss', val_loss.item(), val_iteration)
                writer.add_scalar('Validation/IOU', IOU.item(), val_iteration)
                writer.add_scalar('Validation/Dice', dice.item(), val_iteration)
                writer.add_scalar('Validation/Recall', acc.item(), val_iteration)
                writer.add_scalar('Validation/False_Positive_rate', false_positive_rate.item(), val_iteration)
                writer.add_scalar('Validation/False_Negative_rate', false_negative_rate.item(), val_iteration)

                # set tqdm
                total_valid_loss.append(val_loss.item())
                total_valid_acc.append(acc.item())
                total_valid_IOU.append(IOU.item())
                loop_val.set_description(f'Valid [{epoch}/{epochs}]')
                end_time = time.time()
                loop_val.set_postfix({
                    'm_loss': '{0:1.5f}'.format(mean(total_valid_loss)),
                    'm_acc': '{0:1.5f}'.format(mean(total_valid_acc)),
                    'duration': '{0:1.5f}'.format(end_time - start_time)
                })

            # Save predicted *.nii.gz
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
                # print('turn back from onehot:', np.unique(y))
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

        # Reset timer
        end_time = time.time()
        start_time = end_time

        # log compare train and validation
        writer.add_scalars('Compare/Loss', {'train_loss': mean(total_train_loss),
                                            'valid_train': mean(total_valid_loss)}, epoch)
        writer.add_scalars('Compare/IOU', {'train_IOU': mean(total_train_IOU),
                                           'valid_IOU': mean(total_valid_IOU)}, epoch)

    writer.close()


def test():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(hp.inference_dir, exist_ok=True)

    if hp.mode == '2d':
        if hp.model_name == 'SegNet':
            from models.two_d.segnet2 import SegNet
            model = SegNet(n_init_features=hp.in_class, num_classes=hp.out_class)

        elif hp.model_name == 'UNet':
            from models.two_d.unet import Unet
            model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        elif hp.model_name == 'MiniSeg':
            from models.two_d.miniseg import MiniSeg
            model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        elif hp.model_name == 'PSPNet':
            from models.two_d.pspnet import PSPNet
            model = PSPNet(in_class=hp.in_class, n_classes=hp.out_class)

        elif hp.model_name == 'AttUNet':
            # ATTU_Net 512*512*1
            from models.two_d.attunet import AttU_Net
            model = AttU_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'R2UNet':
            # R2U_Net 512*512*1
            from models.two_d.R2U_Net import R2U_Net
            model = R2U_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'R2AttUNet':
            # R2AttU_Net 265*256*1
            from models.two_d.R2AttU_Net import R2AttU_Net
            model = R2AttU_Net(img_ch=hp.in_class, output_ch=hp.out_class)

        elif hp.model_name == 'DeepLabv3':
            from models.two_d.deeplab2 import DeepLabv3_plus
            model = DeepLabv3_plus(nInputChannels=hp.in_class, n_classes=hp.out_class)

        elif hp.model_name == 'UNetpp':
            from models.two_d.unetpp import ResNet34UnetPlus
            model = ResNet34UnetPlus(num_channels=hp.in_class, num_class=hp.out_class)

        else:
            print('ERROR: No such model')
        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet  # 报错
        # model = SegNet(input_nbr=hp.in_class, label_nbr=hp.out_class)

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

    model = torch.nn.DataParallel(model, device_ids=devices, output_device=[1])

    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                      map_location=lambda storage, loc: storage)
    epoch = ckpt["epoch"]

    model.load_state_dict(ckpt["model"])

    model.to(device)

    full_dataset = MedData_train(hp.source_test_dir, hp.label_test_dir)
    val_loader = DataLoader(full_dataset.dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=hp.num_workers)
    print('ValSet Total Number:', len(full_dataset.dataset))
    print('Data Loaded! Prepare to test......\n')

    model.eval()

    with torch.no_grad():
        loop_test = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, batch in loop_test:
            x = batch['source']['data']
            y = batch['label']['data']

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)
            # x [BS,1,880,880,1]
            # y [BS,1,880,880,1]

            # print('before turn 2 onehot:', np.unique(np.array(y)))
            # y = onehot.mask2onehot(y, hp.out_classlist)
            x = x.type(torch.FloatTensor).to(device)
            y = torch.FloatTensor(y).to(device)
            # y [BS,18,880,880,1]

            outputs = model(x)
            outputs = torch.sigmoid(outputs)

            if hp.mode == '2d':
                x = x.unsqueeze(4)
                y = y.unsqueeze(4)
                outputs = outputs.unsqueeze(4)

            x = x[0].cpu().detach().numpy()
            y = y[0].cpu().detach().numpy()
            # y = y[np.newaxis, :, :, :, :]
            outputs = outputs[0].cpu().detach().numpy()
            outputs = outputs[np.newaxis, :, :, :, :]
            affine = batch['source']['affine'][0].numpy()

            # y = onehot.onehot2mask(y)[0]
            # print('turn back from onehot:', np.unique(y))
            outputs = onehot.onehot2mask(outputs)[0]

            # x [1,880,880,1]
            # y [1,880,880,1]
            # outputs [1,880,880,1]
            source_image = torchio.ScalarImage(tensor=x, affine=affine)
            source_image.save(os.path.join(hp.inference_dir, f"test-step-{epoch:04d}-source_" + str(i) + hp.save_arch))
            # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

            label_image = torchio.ScalarImage(tensor=y, affine=affine)
            label_image.save(os.path.join(hp.inference_dir, f"test-step-{epoch:04d}-gt_" + str(i) + hp.save_arch))

            output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
            output_image.save(os.path.join(hp.inference_dir, f"test-step-{epoch:04d}-predict_" + str(i) + hp.save_arch))

            loop_test.set_description(f'Epoch_Test ')

    # znorm = ZNormalization()
    #
    # if hp.mode == '3d':
    #     patch_overlap = hp.patch_overlap
    #     patch_size = hp.patch_size
    # elif hp.mode == '2d':
    #     patch_overlap = hp.patch_overlap
    #     patch_size = hp.patch_size
    #
    # for i, subj in enumerate(test_dataset.subjects):
    #     subj = znorm(subj)
    #     grid_sampler = torchio.inference.GridSampler(
    #         subj,
    #         patch_size,
    #         patch_overlap,
    #     )
    #
    #     patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=16)
    #     aggregator = torchio.inference.GridAggregator(grid_sampler)
    #     aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
    #     model.eval()
    #     with torch.no_grad():
    #         for patches_batch in tqdm(patch_loader):
    #
    #             input_tensor = patches_batch['source'][torchio.DATA].to(device)
    #             locations = patches_batch[torchio.LOCATION]
    #
    #             if hp.mode == '2d':
    #                 input_tensor = input_tensor.squeeze(4)
    #             outputs = model(input_tensor)
    #
    #             if hp.mode == '2d':
    #                 outputs = outputs.unsqueeze(4)
    #             logits = torch.sigmoid(outputs)
    #
    #             labels = logits.clone()
    #             labels[labels > 0.5] = 1
    #             labels[labels <= 0.5] = 0
    #
    #             aggregator.add_batch(logits, locations)
    #             aggregator_1.add_batch(labels, locations)
    #     output_tensor = aggregator.get_output_tensor()
    #     output_tensor_1 = aggregator_1.get_output_tensor()
    #
    #     affine = subj['source']['affine']
    #
    #     label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
    #     label_image.save(os.path.join(output_dir_test, f"{str(i)}-result_float" + hp.save_arch))
    #
    #     # f"{str(i):04d}-result_float.mhd"
    #
    #     output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
    #     output_image.save(os.path.join(output_dir_test, f"{str(i)}-result_int" + hp.save_arch))


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings('ignore')

    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
