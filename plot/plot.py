import os, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patheffects as path_effects


def mean(list):
    """计算平均值"""
    if not len(list):
        return 0
    return sum(list) / len(list)


# 平滑函数的作用是将每一个点的值变为 上一个节点*0.8+当前节点*0.2
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    smoothed_points = np.array(smoothed_points)
    return smoothed_points


def get_modelnames():
    ignore = ['plot.py', 'model_para.json', 'metrics_model.csv', 'README.md', 'Result', 'R2UNet', 'SegNet']
    model_names = os.listdir('./')
    for i in ignore:
        if i in model_names:
            model_names.remove(i)
    return model_names


def get_plotxy(mod, model_name, x, y, max_epoch, smoothed=0.9, origin=False):
    with open('./model_para.json', 'r') as f:
        model_para = f.read()
        if mod == 'train':
            model_para = json.loads(model_para)['train']
        else:
            model_para = json.loads(model_para)['valid']

    plot_y = []
    ep_metrics = []
    epoch = 1
    for i in range(len(x)):
        if epoch > max_epoch:
            break
        if x[i] <= epoch * model_para[model_name]:
            ep_metrics.append(y[i])
        else:
            plot_y.append(mean(ep_metrics))
            ep_metrics = []
            ep_metrics.append(y[i])
            epoch += 1

    if ep_metrics != [] and x[i] == epoch * model_para[model_name]:
        ep_metrics.append(mean(ep_metrics))

    plot_x = [i for i in range(len(plot_y))]
    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)
    # original
    if origin:
        return epoch, plot_x, plot_y
    # smoothed
    plot_y = smooth_curve(plot_y, factor=smoothed)

    return epoch, plot_x, plot_y


def get_color(id):
    color_map = ['violet', 'neon green', 'muted blue', 'dark coral', 'bluegrey', 'bright pink', 'gold',
                 'ruby', 'pumpkin orange', 'wine', 'azul', 'iris', 'darkish red', 'turquoise',
                 'royal purple', 'purplish blue', 'indian red', 'lavender', 'gunmetal', 'saffron', 'warm blue',
                 'salmon']
    return color_map[id]


def set_plt(x_tick, y_tick):
    # ax = plt.gca()
    # milocx = plt.MultipleLocator(10)
    # ax.xaxis.set_minor_locator(milocx)
    # milocy = plt.MultipleLocator(0.1)
    # ax.yaxis.set_minor_locator(milocy)

    # ax.xaxis.set_ticks_position('bottom')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['left'].set_color('gray')
    # ax.spines['bottom'].set_color('gray')
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')

    # 设置背景色
    # ax.patch.set_facecolor('gray')
    # ax.patch.set_alpha(0.1)

    # 坐标轴数字
    x_ticks = np.arange(0, x_tick + 10, 20)
    plt.xticks(x_ticks)
    y_ticks = np.arange(0, y_tick, 0.1)
    plt.yticks(y_ticks)


def plot_every_metrics(mod, max_epoch, smoothed=0.9, linewidth=1, origin=False):
    train_metrics = ['_loss', '_dice', '_iou', '_fnr', '_fpr', '_recall']
    valid_metrics = ['_Val_Loss', '_Dice', '_IOU', '_False_Negative_rate', '_False_Positive_rate', '_Recall']
    total_metrics = train_metrics if mod == 'train' else valid_metrics

    model_names = get_modelnames()

    # 记录每个model的每个评价指标，每一行为一个记录，记录一个model
    metrics_model = [[i] for i in model_names]

    for metrics in total_metrics:
        plt.figure()
        # plt.figure(dpi=150, figsize=(24, 8))
        for model_name in model_names:
            if mod == 'train':
                csv_name = os.path.join(model_name, model_name + metrics + '.csv')  # train
            else:
                csv_name = os.path.join(model_name, 'run-.-tag-Validation' + metrics + '.csv')  # valid
            if os.path.exists(csv_name):
                data = pd.read_csv(csv_name)
            else:
                print(f'Warning: {csv_name} not exist')

            # original
            if origin:
                epoch, plot_x, plot_y = get_plotxy(mod=mod, model_name=model_name, x=data['Step'], y=data['Value'],
                                                   max_epoch=max_epoch, smoothed=smoothed, origin=origin)
                plt.plot(plot_x, plot_y, linewidth=linewidth, alpha=0.2, label=model_name + 'origin',
                         c=seaborn.xkcd_rgb[get_color(model_names.index(model_name))])
            # smoothed
            epoch, plot_x, plot_y = get_plotxy(mod=mod, model_name=model_name, x=data['Step'], y=data['Value'],
                                               max_epoch=max_epoch, smoothed=smoothed)
            plt.plot(plot_x, plot_y, linewidth=linewidth, label=model_name,
                     # path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()],
                     c=seaborn.xkcd_rgb[get_color(model_names.index(model_name))])
            # # 标注最大值的点
            # max_xy = plot_y.index(max(plot_y)), max(plot_y)
            # plt.plot(plot_y.index(max(plot_y)), max(plot_y), marker='o', markersize=5, c=seaborn.xkcd_rgb[color])
            # plt.annotate(round(max(plot_y), 4), xy=(max_xy), xytext=(max_xy[0] - 3, max_xy[1] + 0.01), c=seaborn.xkcd_rgb[color])

            # 记录该评价指标的最大值
            metrics_model[model_names.index(model_name)].append(max(plot_y))

        set_plt(x_tick=max_epoch, y_tick=1.1)

        # 画图区域
        plt.xlim(0, max_epoch + 10)
        # ['_loss', '_dice', '_iou', '_fnr', '_fpr', '_recall']
        # '_dice', '_iou', '_recall'
        if total_metrics.index(metrics) in [1, 2, 5]:
            plt.ylim(0.3, 1)
        # '_fpr'
        elif total_metrics.index(metrics) in [4]:
            plt.ylim(0, 0.05)
            y_ticks = np.arange(0, 0.055, 0.005)
            plt.yticks(y_ticks)
            milocy = plt.MultipleLocator(0.005)
            ax = plt.gca()
            ax.yaxis.set_minor_locator(milocy)
        # '_fnr', 'loss'
        else:
            plt.ylim(0, 1)

        # 设置坐标轴名称
        plt.xlabel('Epoch', fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)
        plt.ylabel(metrics[1:].title(), fontdict={'weight': 'normal'}, fontsize=12, labelpad=6)
        # 设置标题 fontdict={'weight': 'normal'}
        # plt.suptitle(metrics[1:], fontdict={'family': 'Times New Roman'}, fontsize=25)

        # 设置网格
        plt.grid(linewidth=0.7, color='gray', alpha=0.3, which='both', linestyle='--')
        plt.legend()
        # 设置图片大小
        ax = plt.gcf()
        # ax.set_size_inches(12, 5.25)
        ax.set_size_inches(8, 8)
        plt.savefig(os.path.join('Result', metrics[1:] + '.png'), bbox_inches='tight', pad_inches=0.0, transparent=True)
        print(f'Saved to {metrics[1:]}.png')
        plt.close()
        # plt.show()

    # 每个model的每个评价指标写入csv
    data = pd.DataFrame(metrics_model, columns=(['model_name'] + total_metrics))
    data.to_csv('metrics_model.csv')


def plot_every_model(max_epoch=300, smoothed=0.8, linewidth=1, origin=False):
    model_names = get_modelnames()

    for model_name in model_names:
        plt.figure()
        csv_names = [model_name + '_loss', 'run-.-tag-Validation_Val_Loss', model_name + '_dice',
                     'run-.-tag-Validation_Dice']
        for id in range(4):
            csv_name = os.path.join(model_name, csv_names[id] + '.csv')  # valid
            if os.path.exists(csv_name):
                data = pd.read_csv(csv_name)
            else:
                print(f'Warning: {csv_name} not exist')
            if id in [0, 2]:
                # original
                if origin:
                    epoch, plot_x, plot_y = get_plotxy(mod='train', model_name=model_name, x=data['Step'],
                                                       y=data['Value'], max_epoch=max_epoch, smoothed=0.3,
                                                       origin=origin)
                    plt.plot(plot_x, plot_y, linewidth=linewidth, alpha=0.3, label=None)
                # smoothed
                epoch, plot_x, plot_y = get_plotxy(mod='train', model_name=model_name, x=data['Step'], y=data['Value'],
                                                   max_epoch=max_epoch, smoothed=smoothed)
                plt.plot(plot_x, plot_y, linewidth=linewidth, label='Train_' + csv_name[-8:-4].title())
            else:
                # original
                if origin:
                    epoch, plot_x, plot_y = get_plotxy(mod='valid', model_name=model_name, x=data['Step'],
                                                       y=data['Value'], max_epoch=max_epoch, smoothed=0.3,
                                                       origin=origin)
                    plt.plot(plot_x, plot_y, linewidth=linewidth, alpha=0.3, label=None)
                # smoothed
                epoch, plot_x, plot_y = get_plotxy(mod='valid', model_name=model_name, x=data['Step'], y=data['Value'],
                                                   max_epoch=max_epoch, smoothed=smoothed)
                plt.plot(plot_x, plot_y, linewidth=linewidth, label='Valid_' + csv_name[-8:-4].title())

        set_plt(x_tick=max_epoch, y_tick=1.1)

        # 画图区域
        plt.xlim(0, epoch + 10)
        plt.ylim(0.05, 0.95)

        # 设置坐标轴名称
        plt.xlabel('Epoch', fontdict={'weight': 'normal'}, fontsize=16)
        plt.ylabel('Loss', fontdict={'weight': 'normal'}, fontsize=16)
        plt.suptitle(model_name, fontdict={'family': 'Times New Roman'}, fontsize=25)

        # 设置网格
        plt.grid(linewidth=0.7, color='gray', alpha=0.3, which='both', linestyle='--')
        plt.legend()
        # 设置图片的大小
        ax = plt.gcf()
        # ax.set_size_inches(12, 6.75)
        ax.set_size_inches(6, 6)
        # 设置双y轴
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_ylim(0.05, 0.95)
        ax2.set_ylabel('Dice', fontdict={'weight': 'normal'}, fontsize=18)
        # 保存
        plt.savefig(os.path.join('Result', model_name + '.png'), bbox_inches='tight', pad_inches=0.0, transparent=True)
        print(f'Saved to {model_name}.png')
        plt.close()
        # plt.show()


def plot_pic():
    plt.figure(figsize=(15, 46))
    ignore = ['InfNet-Res2Net50', 'PraNet-Res2Net50', 'SegNet', 'R2UNet']
    titles = ['(a) Best', '(b) Worst', '(c) Random1', '(d) Random2', '(e) Random3']
    model_names = []
    pic_nums = ['71.png', '448.png', '134.png', '150.png', '194.png']
    pics = os.listdir('../pic')
    for i in pics:
        title = i.split('_')
        if title[0] not in model_names:
            model_names.append(title[0])
    for i in ignore:
        if i in model_names:
            model_names.remove(i)

    if 'Groundtruth' in model_names:
        model_names.remove('Groundtruth')
        model_names = ['Groundtruth'] + model_names
    else:
        print('Warning no Groundtruth picture')
    if 'Image' in model_names:
        model_names.remove('Image')
        model_names = ['Image'] + model_names
    else:
        print('Warning no Image picture')

    id = 1
    for row in range(len(model_names)):
        for column in range(len(pic_nums)):
            img = Image.open(os.path.join('../pic', model_names[row] + '_' + pic_nums[column]))
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            plt.subplot(len(model_names), len(pic_nums), id)
            if column == 0:
                plt.ylabel(model_names[row], rotation=0, fontsize=20, labelpad=80)
            if row == 0:
                plt.title(titles[column], fontdict={'family': 'Times New Roman'}, fontsize=25)
            id += 1
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        print(f'{row} done')

    plt.gcf().subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)
    plt.savefig('../seg_result.png', bbox_inches='tight', pad_inches=0.0, transparent=True)
    # plt.show()


if __name__ == '__main__':
    # 设置xtick和ytick的方向：in、out、inout
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # 画曲线
    os.makedirs('./Result', exist_ok=True)
    plot_every_metrics(mod='valid', max_epoch=151)
    plot_every_model()

    # 画图片
    plot_pic()
