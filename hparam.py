class hparams:
    train_or_test = 'train'  # 'train' or 'test'
    output_dir = 'logs'
    inference_dir = 'results'
    aug = True
    latest_checkpoint_file = 'checkpoint_0010.pt'
    # total_epochs = 5000000
    total_epochs = 200
    epochs_per_checkpoint = 1
    batch_size = 1
    ckpt = None  # 用来断点继续训练，例如:'checkpoint_100.pt'
    val_split = 0.2  # 验证集/数据集总数，一般训练集:验证集 = 4:1
    unused_split = 0.0  # 不加载的数据/数据集总数，若train_or_test = 'train' 则此次不加载
    # 剩下的就是训练集的个数
    init_lr = 0.002
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

    # for test
    patch_overlap = 4, 4, 0  # if 3D: 4,4,4

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'

    # source_train_dir = './dataset/testslice/source'
    # label_train_dir = './dataset/testslice/label/'
    source_train_dir = './dataset/slice_train/source/'
    label_train_dir = './dataset/slice_train/label/'
    source_test_dir = './dataset/test/source/'
    label_test_dir = './dataset/test/label/'
