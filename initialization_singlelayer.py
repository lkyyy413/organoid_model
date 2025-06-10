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
    
    def create_cells_hex(self, num_cells_x=5, num_cells_y=5, num_cells_z=5, 
                    radius=1.0, height=0.5):
        """
        生成六棱柱阵列的顶点坐标 (num_cells_total, 12, 3)
        
        参数:
            num_cells_x, num_cells_y: 水平方向的六棱柱数量
            num_cells_z: 垂直方向的层数
            radius: 六棱柱底面外接圆半径
            height: 六棱柱高度
        """
        num_cells_total = num_cells_x * num_cells_y * num_cells_z
        vertices = torch.zeros(num_cells_total, 12, 3, device=self.device)
        
        # 定义单个六棱柱的12个顶点（局部坐标，以中心为原点）
        # 底面6个顶点
        bottom_vertices = []
        # 顶面6个顶点
        top_vertices = []
        
        for i in range(6):
            angle = math.pi / 3 * i  # 60度一个顶点
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            bottom_vertices.append([x, y, -height/2])
            top_vertices.append([x, y, height/2])
        
        local_vertices = torch.tensor(bottom_vertices + top_vertices, device=self.device)
        
        # 在全局坐标系中排列六棱柱（蜂窝状排列）
        idx = 0
        hex_spacing_x = radius * 1.5  # 水平间距
        hex_spacing_y = radius * math.sqrt(3)  # 垂直间距
        
        for z in range(num_cells_z):
            for y in range(num_cells_y):
                for x in range(num_cells_x):
                    # 蜂窝状排列的偏移
                    offset_x = hex_spacing_x * x
                    offset_y = hex_spacing_y * y
                    if x % 2 == 1:  # 奇数行需要垂直偏移
                        offset_y += hex_spacing_y / 2
                    
                    center = torch.tensor([
                        offset_x,
                        offset_y,
                        z * height  # 垂直方向间距
                    ], device=self.device)
                    
                    vertices[idx] = local_vertices + center
                    idx += 1
        
        hex_indices = torch.arange(num_cells_total, device=self.device)
        hex_dict = {i.item(): vertices[i] for i in hex_indices}
        
        return hex_dict

