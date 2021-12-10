## 如何使用nnUNet跑二维MRI
### 教程参考
* （四：2020.07.28）nnUNet最舒服的训练教程（让我的奶奶也会用nnUNet（上））（21.04.20更新）
`https://blog.csdn.net/weixin_42061636/article/details/107623757`
* 保姆级教程：nnUnet在2维图像的训练和测试
`https://blog.csdn.net/minervazhaojie/article/details/112061000`
* （十七：2020.09.10）nnUNet最全问题收录（9.10更新）
`https://blog.csdn.net/weixin_42061636/article/details/108520695`

### 步骤
* 安装hiddenlayer（用来生成什么网络拓扑图？管他呢，装吧）
```bash
pip install --upgrade git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer
```
* 安装nnUNet
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
* 找到.bashrc,写入
```bash
export nnUNet_raw_data_base="/home/qiao/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/home/qiao/nnUNetFrame/DATASET/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/qiao/nnUNetFrame/DATASET/nnUNet_trained_models"
```
* 更新刚才写入的.bashrc
```bash
source .bashrc
```
* 创建一堆文件夹
```bash
nnUNetFrame
    ├──DATASET
    ├── dataset.json
    ├── imagesTr
    │   ├── Case100_10_0000.nii.gz
    │   ├── Case100_11_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── Case161_10_0000.nii.gz
    │   ├── Case161_11_0000.nii.gz
    │   ├── ...
    └── labelsTr
```
* 下载开源数据集(建议04,最小)，我使用自己的，因此需要照着改
`https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2`
    * 修改labels,numTraining,numTest
    * Task后面的id,必须是三位数
    * 在dataset.json中修改:
        * `"training":[{"image":"./imagesTr/Case100_1.nii.gz","label":"./labelsTr/Case100_1.nii.gz"},...]`
        * `"test":["./imagesTs/Case161_1.nii.gz",...]`
    * 注意:
        * 对于文件:imagesTr和imagesTs需要加上`_0000.nii.gz`,labelsTr不需要加，但是要和imagesTr名字对应
        * 对于dataset.json:都不需要加`_0000`
```bash
nnUNet_raw_data_base/nnUNet_raw_data/Task058_Spine
├── dataset.json
├── imagesTr
│   ├── Case100_10_0000.nii.gz
│   ├── Case100_11_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── Case161_10_0000.nii.gz
│   ├── Case161_11_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── Case100_1.nii.gz
    ├── Case100_10.nii.gz
    ├── ...
```
* verify数据集 & Run
```bash
nnUNet_plan_and_preprocess -t 058 -pl3d None --verify_dataset_integrity
nnUNet_train 2d nnUNetTrainerV2 058 all
nohup nnUNet_train 2d nnUNetTrainerV2 058 all > runlog.txt 2>&1 &
```