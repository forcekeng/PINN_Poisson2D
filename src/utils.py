from mindspore.train.serialization import load_checkpoint, load_param_into_net


def load_paramters_into_net(param_path, net):
    """载入训练好的参数"""
    param_dict = load_checkpoint(param_path)
    convert_ckpt_dict = {}
    for _, param in net.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(net, convert_ckpt_dict)
    print("Load parameters finished!")