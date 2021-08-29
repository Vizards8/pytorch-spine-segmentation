class hparams:
    train_or_test = 'train'  # 'train' or 'test'
    # total_epochs = 5000000
    total_epochs = 300
    epochs_per_checkpoint = 50  # 已经添加了每轮都保存，这个不用管
    small_sample = False  # 使用较少数量的样本，用于调试，启用下面的小数
    small_sample_split = 0.01  # small_sample = True时生效，本次采用的数据量/总数据量
    # 剩下的就是训练集的个数
    batch_size = 1  # 多卡训练，例如：2张3090,2*4=8
    model_name = 'UNet'  # 'UNet' 'SegNet' 'MiniSeg' 'PSPNet' 'AttUNet' 'R2UNet' 'R2AttUNet' 'DeepLabv3' 'UNetpp'

    output_dir = 'logs'
    inference_dir = 'results'
    aug = True
    latest_checkpoint_file = 'checkpoint_latest.pt'
    gpu_nums = 1  # 多卡训练设置有几张GPU
    ckpt = None  # 用来断点继续训练，例如:'checkpoint_100.pt'
    init_lr = 0.005
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d'  # '2d or '3d'
    in_class = 1
    out_class = 20
    out_classlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    use_queue = False
    samples_per_volume = 5
    crop_or_pad_size = 880, 880, 1
    # if 2D: 256,256,1 if 3D: 256,256,256
    patch_size = 512, 512, 1
    # if 2D: 128,128,1 if 3D: 128,128,128
    num_workers = 0  # cpu电脑将此改为0，我觉得好像没啥区别
    fold_arch = '*.nii.gz'
    save_arch = '.nii.gz'
    # source_train_dir = './dataset/testslice/source'
    # label_train_dir = './dataset/testslice/label/'
    source_train_dir = './dataset/slice_train/source/'
    label_train_dir = './dataset/slice_train/label/'
    source_test_dir = './dataset/slice_test/source/'
    label_test_dir = './dataset/slice_test/label/'
