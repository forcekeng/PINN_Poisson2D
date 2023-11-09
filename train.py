import os
import time
import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.solver import Solver
from mindelec.common import L2
from mindelec.loss import Constraints
from mindelec.architecture import MultiScaleFCCell

from src.config import poisson_2d_config
from src.poisson2d import Poisson2D
from src.dataset import generate_train_dataset
from src.callback import TimeMonitor, SaveCkptMonitor, EpochLossMonitor
from src.utils import load_paramters_into_net

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend")

# 设置随机数种子
ms.common.set_seed(123456)


def plot_train_loss(loss_cb, geom="", save_dir="output/"):
    """绘制训练损失"""
    import numpy as np
    import matplotlib.pyplot as plt
    loss = loss_cb.loss_record
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(loss)+1), loss, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel(f'{geom} loss')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}loss_{geom}.png', bbox_inches='tight', dpi=150)
    plt.show()
    

def train(geom='rectangle', param_path=""):
    print(f'Geometry is {geom}.')
    # define trainable network
    config = poisson_2d_config
    net = MultiScaleFCCell(in_channel=config["in_channel"], 
                       out_channel=config["out_channel"], 
                       layers=config["layers"],
                       neurons=config["neurons"])
    if param_path:
        load_paramters_into_net(param_path, net)
    net.to_float(ms.dtype.float16)
    
    # define dataset
    train_dset = generate_train_dataset(geom)
    train_data_loader = train_dset.create_dataset(batch_size=poisson_2d_config.batch_size,
                       shuffle=True)
        
    # define problem 
    train_prob_dict = {geom: Poisson2D(net=net,
                                       domain_name=geom+"_domain_points",
                                       bc_name=geom+"_BC_points",
                                      )
                      }
    
    # define constraints
    train_constraints = Constraints(train_dset, train_prob_dict)
    
    # optimizer
    optimizer = nn.Adagrad(net.trainable_params(), learning_rate=poisson_2d_config.lr)
    
    # solver
    solver = Solver(net,
                    optimizer=optimizer,
                    mode="PINNs",
                    train_constraints=train_constraints,
                    metrics={'l2': L2(), 'distance': nn.MAE()},
                    amp_level="O3"
                   )
    
    # callback
    time_cb = TimeMonitor()
    loss_cb = EpochLossMonitor()
    save_cb = SaveCkptMonitor(comment=geom)
    
    # begin train
    print("<================= Begin Trainning =================>")
    tstart = time.time()
    solver.train(epoch=poisson_2d_config.epochs, 
                 train_dataset=train_data_loader,
                 callbacks=[loss_cb, time_cb, save_cb]
                )
    
    tend = time.time()
    print(f"Train task total spends {(tend - tstart)/60:.3f} minutes.\n")
    plot_train_loss(loss_cb, geom=geom)
    print("<================= End Trainning =================>\n")
    
    

if __name__ == "__main__":
    """train process"""
    # geom 可选 {'disk', 'rectangle', 'polygon', 'triangle'}
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
    
    train(args.geom, args.weight)
    