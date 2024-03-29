"""
call back functions
"""
import os
import time
import mindspore as ms
from mindspore.train.callback import Callback


class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
    data_size (int): Iteration steps to run one epoch of the whole dataset.
    """
    def __init__(self, data_size=None):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.per_step_time = 0

    def epoch_begin(self, run_context):
        """
        Set begin time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        run_context.original_args()
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        self.per_step_time = epoch_seconds / step_size
        print("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, self.per_step_time), flush=True)

    def get_step_time(self):
        return self.per_step_time

    
class EpochLossMonitor(Callback):
    """每个epoch打印一次loss，并记录训练loss"""
    def __init__(self, max_len=-1):
        self.max_len = max_len # 最大记录长度，-1记录所有
        self.loss_record = []
    
    def epoch_end(self, run_context):
        """定义step结束时的执行操作"""
        cb_params = run_context.original_args()
        # 如果当前损失值小于设定的阈值就停止训练
        epoch = cb_params.cur_epoch_num
        cur_loss = cb_params.net_outputs.asnumpy() # 获取当前损失值
        self.loss_record.append(cur_loss)
        
        if self.max_len > 0:
            self.loss_record = self.loss_record[-self.max_len:]
            
        print(f"Epoch: {epoch}, loss: {cur_loss:.5f}.")
    

class SaveCkptMonitor(Callback):
    """定义保存ckpt文件的回调接口，保存损失比上一次小的模型"""
    
    def __init__(self, loss=100.0, save_dir="./checkpoints/", comment=""):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss # 低于此损失才开始保存模型
        self.save_dir = save_dir
        self.comment = comment
        
    def epoch_end(self, run_context):
        """定义step结束时的执行操作"""
        cb_params = run_context.original_args()
        # 如果当前损失值小于设定的阈值就停止训练
        epoch = cb_params.cur_epoch_num
        cur_loss = cb_params.net_outputs.asnumpy() # 获取当前损失值
        
        if cur_loss < self.loss:
            self.loss = cur_loss
            # 自定义保存文件名
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                
            file_name = f"{self.comment}_epoch_{epoch}_loss_{cur_loss:.6f}.ckpt"
            file_name = os.path.join(self.save_dir, file_name)
            # 保存网络模型
            ms.save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print(f"Current epoch: {epoch}, loss: {cur_loss}, saved checkpoint.")
