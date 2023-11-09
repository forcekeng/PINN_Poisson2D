import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.common import L2
from mindelec.common import PI
from mindelec.architecture import MultiScaleFCCell

from src.config import sampling_config, poisson_2d_config
from src.dataset import generate_test_dataset
from src.utils import load_paramters_into_net
from geom.geometry_2d import Disk, Polygon, Rectangle, Triangle


GEOM_SUPPORT = ["disk", "polygon", "rectangle", "triangle"]

def get_mask(geom, points):
    r"""
    Create dataset.

    Args:
        name (str): name of the disk.
    
    Return:
        dataset: Dataset
    """
    assert geom in GEOM_SUPPORT, f"The supportted goem_name is {GEOM_SUPPORT} but got {geom_name}."
    
    if geom.lower() == "disk":
        sample_config = sampling_config.disk
        def_config = poisson_2d_config.geom_define.disk
        space = Disk(def_config.name, 
                     def_config.center, def_config.radius)
        return space._inside(points)
    
    elif geom.lower() == "polygon":
        sample_config = sampling_config.polygon
        def_config = poisson_2d_config.geom_define.polygon
        space = Polygon(def_config.name, def_config.vertex)
        return space._inside(points)

    elif geom.lower() == "rectangle":
        sample_config = sampling_config.rectangle
        def_config = poisson_2d_config.geom_define.rectangle
        space = Rectangle(def_config.name, 
                          def_config.coord_min, def_config.coord_max)
        return space._inside(points)
        
    elif geom.lower() == "triangle":
        sample_config = sampling_config.triangle
        def_config = poisson_2d_config.geom_define.triangle
        
        space = Triangle(def_config.name, def_config.vertex)
        return space._inside(points)
    

def grid_plot(net, geom, save_dir='output/'):
    """网格绘图，显示结果"""
    size = poisson_2d_config["plot_size"]
    wn = poisson_2d_config["wave_number"]
    k0 = 2.0 * PI * wn
    
    xx = np.linspace(-1, 1, size)
    yy = np.linspace(-1, 1, size)
    xss, yss = np.meshgrid(xx, yy)
    xs, ys = xss.reshape((-1,1)), yss.reshape((-1, 1))
    points = np.c_[xs, ys]
    mask = get_mask(geom, points)
    mask = mask.astype(np.float32).reshape(xss.shape)
    
    test_data = ms.Tensor(points, ms.dtype.float32)
    test_pred = net(test_data).asnumpy().reshape(xss.shape)
    test_pred = test_pred * mask
    test_true = 1.0/2.0/k0/k0 * (np.sin(k0*xs) * np.sin(k0*ys)).reshape(xss.shape) * mask
    diff = np.abs(test_pred - test_true)
    # 原始图片向下为y轴正方向，上下翻转，使得向上为y轴正方向
    test_pred = np.flipud(test_pred)
    test_true = np.flipud(test_true)
    diff = np.flipud(diff)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax = ax1.imshow(test_pred, cmap='jet')
    ax1.set_title(f'{geom} prediction')
    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax1.set_xticks([0, size-1], [-1, 1])
    ax1.set_yticks([0, size-1], [1, -1])
    
    ax = ax2.imshow(test_true, cmap='jet')
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_title(f'{geom} ground truth')
    ax2.set_xticks([0, size-1], [-1, 1])
    ax2.set_yticks([0, size-1], [1, -1])
    
    fig.colorbar(ax, ax=[ax1, ax2], shrink=0.5)
    ax = ax3.imshow(diff, cmap='binary')
    ax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax3.set_xticks([0, size-1], [-1, 1])
    ax3.set_yticks([0, size-1], [1, -1])
    ax3.set_title(f'difference')
    fig.colorbar(ax, ax=[ax3], shrink=0.5)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}{geom}_eval_result.png', bbox_inches='tight', dpi=150)


def analytic_solution(xy):
    """解析解
    xy: ms.Tensor: shape=(n, 2): 两列坐标，分别为x和y
    """
    k0 = 2.0 * PI * poisson_2d_config["wave_number"]
    res = 1.0 / k0**2 /2.0 * ops.sin(k0 * xy[:, 0:1]) * ops.sin(k0 * xy[:, 1:2])
    return res


def get_l2_relative_loss(pred, label):
    """计算L2损失
    pred: np.ndarray: shape=(n,): 预测值
    label: np.ndarray: shape=(n,): 真实值
    """
    diff = np.abs(pred - label)
    rel_diff = diff / label
    loss = np.sqrt(np.mean(rel_diff**2))
    error = diff.max() / np.abs(label).max()
    return loss, error

    
def evaluate(geom, param_path=""):
    """评估
    geom: in {'disk', 'rectangle', 'polygon', 'triangle'}
    weight: 训练好的参数路径
    """
    tstart = time.time()
    print("<=================== Begin Evaluation ===================>")
    print(f"Geometry: {geom}")
    # 定义网络
    net = MultiScaleFCCell(in_channel=poisson_2d_config["in_channel"], 
                   out_channel=poisson_2d_config["out_channel"], 
                   layers=poisson_2d_config["layers"],
                   neurons=poisson_2d_config["neurons"])
    # 加载模型权重
    if param_path == "":
        param_path = poisson_2d_config["param_path"]
    load_paramters_into_net(param_path, net)
    
    grid_plot(net, geom)
    # 生成测试数据
    test_dset = generate_test_dataset(geom)
    test_data_loader = test_dset.create_dataset(batch_size=poisson_2d_config.batch_size,
                                                 shuffle=False, drop_remainder=False)
    # 获取预测值和真实值
    domain_preds = []
    domain_labels = []
    bc_preds = []
    bc_labels = []
    for inputs in test_data_loader:
        domain_pred = net(inputs[0])
        bc_pred = net(inputs[1])
        
        domain_label = analytic_solution(inputs[0])
        bc_label = analytic_solution(inputs[1])
        
        domain_labels.append(domain_label.asnumpy())
        bc_labels.append(bc_label.asnumpy())
        domain_preds.append(domain_pred.asnumpy())
        bc_preds.append(bc_pred.asnumpy())
    
    domain_preds = np.vstack(domain_preds).flatten()
    domain_labels = np.vstack(domain_labels).flatten()
    bc_preds = np.vstack(bc_preds).flatten()
    bc_labels = np.vstack(bc_labels).flatten()
    
    # 计算损失
    domain_loss = get_l2_relative_loss(domain_preds, domain_labels)
    bc_loss = get_l2_relative_loss(bc_preds, bc_labels)
    
    print(f"Domain L2 relative loss = {domain_loss}")
    print(f"Boundary L2 relative loss = {bc_loss}")
    
    tend = time.time()
    print(f"Evaluation task total spends {(tend - tstart):.2f} seconds.")
    print("<=================== End Evaluation ===================>")
    

if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser(description="Train poisson 2D with PINNs.")
    parser.add_argument('-g', '--geom', 
                        type=str, 
                        choices=['disk', 'rectangle', 'polygon', 'triangle'],
                        help="The define geometry.",
                        default='rectangle') 
    parser.add_argument('-w', '--weight',
                        type=str,
                        help="Pretrained parameters path.",
                        default='')
    
    args = parser.parse_args()
    
    evaluate(args.geom, args.weight)
    
