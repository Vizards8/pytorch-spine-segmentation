# Logs
* 8.14 多分类标签问题调整完毕，现输出18分类，效果还行
* 8.15 添加validation
* 8.16 尝试colab，但是爆内存，修改切片大小为660*660，能跑，但是gpu限额
* 8.17 添加读数据进度条
* 8.17 添加test_split参数，可调整每次读取的数据量，不必删数据集
* 8.17 添加跑代码过程中的进度条，优化可读性
* 8.18 添加use_queue参数，可将一张原图切成多张小图读取
* 8.18 修改test
* 8.18 添加postprocess，组合为三维MRI
* 8.19 添加SegNet支持
* 8.20 添加PSPNet支持
* 8.23 添加Using Device，方便调试
* 8.25 添加IOU，Dice，FP，FN，待测试

# Todo
* 切片880*880过大，不合理，需要trick
* 边缘切片没有18个分类，效果肯定不好，是否影响整体模型
* RandomElasticDeformation()会报warning，为了观感，暂时去除
* 其他模型
* 其他Loss函数
* 其他评价指标
* 有的切片没有18个分类，导致one-hot存在全0tensor，tp = 0, fn = 0, 正确率虚高

# Need to include
* number of params
* evaluate
    * IOU
    * Dice
    * pixel acc
* model
    * UNet
    * FCN
    * UNet++
    * PSPNet
    * DeepLabv3