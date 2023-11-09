from easydict import EasyDict as ed

# 矩形、圆、三角形、五边形
sampling_config = ed({
    "disk": ed({                         # disk 采样设置
        'domain': ed({                   # 内部采样
            'random_sampling': True,     # 随机采样
            'size': 65536,               # 采样点数
        }),
        'BC': ed({                       # 边界采样
            'random_sampling': True,     # 随机采样
            'size': 8192,                # 采样点数
        })
    }),
    
    "polygon": ed({                      # 多边形采样
        'domain': ed({
            'random_sampling': True,
            'size': 65536,
        }),
        'BC': ed({
            'random_sampling': True,
            'size': 8192,
        }),
    }),
    
    "triangle": ed({                     # 三角形采样
        'domain': ed({
            'random_sampling': True,
            'size': 65536,
        }),
        'BC': ed({
            'random_sampling': True,
            'size': 8192,
        }),
    }),
    
    "rectangle": ed({                    # 矩形采样
        'domain': ed({
            'random_sampling': True,
            'size': 65536,
        }),
        'BC': ed({
            'random_sampling': True,
            'size': 8192,
        }),
    }),
})


# config
poisson_2d_config = ed({
    "name": "poisson2d",                            # 任务名，不重要
    "geom_define": ed({                             # 几何体定义
        "disk": ed({                                # 圆形
            "name": "disk",                         # 名字
            "center": [0, 0],                       # 圆心
            "radius": 1,                            # 半径
        }),

        "rectangle": ed({                           # 矩形
            "name": "rectangle", 
            "coord_min": [-1, -0.5],                # 最小坐标
            "coord_max": [1, 0.5],                  # 最大坐标
        }),
        
        "triangle": ed({                            # 三角形
            "name": "triangle",
            "vertex": [[-1, -1], [1, -1], [0, 1]],  # 三个顶点
        }),
        
        "polygon": ed({                             # 多边形
            "name": "polygon",
            "vertex": [[0,1], [0.951, 0.309], 
                       [0.588, -0.809], [-0.588, -0.809],
                       [-0.951, 0.309]],            # 五边形顶点
        }),
    }),
    # 训练参数
    "wave_number": 2,                # 波数，乘以 2*pi 为题目要求的 4*pi
    "epochs": 1000,                  # epoch数量
    "batch_size": 128,               # 训练batch_size
    "lr": 1e-3,                      # 学习率
    "optimizer": "adam",             # 优化器,取 {'adam', 'adagrad'}

    # 神经网络配置
    "in_channel": 2,                 # 输入通道数
    "out_channel": 1,                # 输出通道数
    "layers": 4,                     # 重复次数
    "neurons": 64,                   # 每层的神经元数量
    
    # 测试配置
    "param_path": "checkpoints/xxx.ckpt",  # 训练好的权重，也可通过命令行传参指定
    "plot_size": 201,                # 绘图网格
})
