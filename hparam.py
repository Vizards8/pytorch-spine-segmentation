class hparams:
    train_or_test = 'test'
    output_dir = 'logs'
    inference_dir = 'results'
    aug = True
    latest_checkpoint_file = 'checkpoint_latest.pt'
    # total_epochs = 5000000
    total_epochs = 200
    epochs_per_checkpoint = 10
    batch_size = 2
    ckpt = None
    val_split = 0.2
    test_split = 0.0
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d'  # '2d or '3d'
    in_class = 1
    out_class = 18
    out_classlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]

    use_queue = False
    crop_or_pad_size = 440, 440, 1
    # if 2D: 256,256,1 if 3D: 256,256,256
    patch_size = 880, 880, 1
    # if 2D: 128,128,1 if 3D: 128,128,128
    num_workers = 0

    # for test
    patch_overlap = 4, 4, 0  # if 3D: 4,4,4

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'

    source_train_dir = './dataset/testslice/source'
    label_train_dir = './dataset/testslice/label/'
    # source_train_dir = './dataset/slice_train/source/'
    # label_train_dir = './dataset/slice_train/label/'
    source_test_dir = './dataset/test/source/'
    label_test_dir = './dataset/test/label/'
