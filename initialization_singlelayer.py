import torch
import numpy as np
import math

class Initialization:
    def __init__(self, device='cuda'):
        self.device = device

    def create_cells(self, num_cells_x=5, num_cells_y=5, num_cells_z=5, 
                size_x=1.0, size_y=1.0, size_z=0.5, spacing=1.0):
        """
        生成长方体阵列的顶点坐标 (num_cells_total, 8, 3)
    
        参数:
            num_cells_x, num_cells_y, num_cells_z: 三个方向的细胞数量
            size_x, size_y, size_z: 长方体的长、宽、高
            spacing: 细胞间距（相对于各自边长的比例）
        """
        num_cells_total = num_cells_x * num_cells_y * num_cells_z
        vertices = torch.zeros(num_cells_total, 8, 3, device=self.device)
    
        # 定义单个长方体的8个顶点（局部坐标，以中心为原点）
        half_x = size_x / 2
        half_y = size_y / 2
        half_z = size_z / 2
        local_vertices = torch.tensor([
            [-half_x, -half_y, -half_z],  # 0: 左前下
            [ half_x, -half_y, -half_z],  # 1: 右前下
            [ half_x,  half_y, -half_z],  # 2: 右后下
            [-half_x,  half_y, -half_z],  # 3: 左后下
            [-half_x, -half_y,  half_z],  # 4: 左前上
            [ half_x, -half_y,  half_z],  # 5: 右前上
            [ half_x,  half_y,  half_z],  # 6: 右后上
            [-half_x,  half_y,  half_z]   # 7: 左后上
        ], device=self.device)
    
        # 在全局坐标系中排列长方体
        idx = 0
        for z in range(num_cells_z):
            for y in range(num_cells_y):
                for x in range(num_cells_x):
                    center = torch.tensor([
                        x * (size_x * spacing),
                        y * (size_y * spacing),
                        z * (size_z * spacing)
                    ], device=self.device)
                    vertices[idx] = local_vertices + center
                    idx += 1
        cube_indices = torch.arange(num_cells_total, device=self.device)
        cube_dict = {i.item(): vertices[i] for i in cube_indices}
    
        return cube_dict