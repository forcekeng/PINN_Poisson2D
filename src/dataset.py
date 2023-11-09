# 产生数据集
import numpy as np

from mindelec.data import Dataset
from mindelec.geometry import create_config_from_edict

from geom.geometry_2d import Disk, Polygon, Rectangle, Triangle
from src.config import sampling_config, poisson_2d_config

GEOM_SUPPORT = ["disk", "polygon", "rectangle", "triangle"]

def generate_train_dataset(geom: str) -> Dataset:
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
                     def_config.center, 
                     def_config.radius,
                     sampling_config=create_config_from_edict(sample_config))
        geom_dict = {space: ["domain", "BC"]}
    
    elif geom.lower() == "polygon":
        sample_config = sampling_config.polygon
        def_config = poisson_2d_config.geom_define.polygon
        space = Polygon(def_config.name, 
                        def_config.vertex,
                        sampling_config=create_config_from_edict(sample_config))
        geom_dict = {space: ["domain", "BC"]}

    elif geom.lower() == "rectangle":
        sample_config = sampling_config.rectangle
        def_config = poisson_2d_config.geom_define.rectangle
        space = Rectangle(def_config.name, 
                        def_config.coord_min,
                        def_config.coord_max,
                        sampling_config=create_config_from_edict(sample_config))
        geom_dict = {space: ["domain", "BC"]}
        
    elif geom.lower() == "triangle":
        sample_config = sampling_config.triangle
        def_config = poisson_2d_config.geom_define.triangle
        
        space = Triangle(def_config.name, 
                         def_config.vertex,
                         sampling_config=create_config_from_edict(sample_config))
        geom_dict = {space: ["domain", "BC"]}
        
    dataset = Dataset(geom_dict)
    return dataset


def generate_test_dataset(geom: str)->Dataset:
    """生成测试数据集"""
    return generate_train_dataset(geom)