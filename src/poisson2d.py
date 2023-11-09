
from mindspore import ms_function
from mindspore import Tensor
import mindspore.ops as ops
from mindelec.solver import Problem
from mindelec.operators import SecondOrderGrad
from mindelec.common import PI

from src.config import poisson_2d_config

class Poisson2D(Problem):
    """2D Poisson equation"""
    def __init__(self, net, domain_name, bc_name):
        super(Poisson2D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.net = net
        self.type = "Equation"
        self.wave_number = poisson_2d_config["wave_number"]
        self.k0 = Tensor(2.0 * PI * self.wave_number)
        # 二阶梯度算子
        self.grad_xx = SecondOrderGrad(net, 0, 0, 0)
        self.grad_yy = SecondOrderGrad(net, 1, 1, 0)
        
        
    @ms_function
    def governing_equation(self, *output, **kwargs):
        """governing equation"""
        data = kwargs[self.domain_name]
        x, y = data[:, 0:1], data[:, 1:2]
        u_xx = self.grad_xx(data)
        u_yy = self.grad_yy(data)
        return 10.0 * (u_xx + u_yy + ops.sin(self.k0*x) * ops.sin(self.k0*y))
    
    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """boundary equation"""
        u = output[0]
        data = kwargs[self.bc_name]
        x, y = data[:, 0:1], data[:, 1:2]
        label = 1.0/self.k0/self.k0/2 * ops.sin(self.k0 * x) * ops.sin(self.k0 * y)
        return 100.0*(u - label)
        

