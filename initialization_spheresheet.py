import torch
import numpy as np
import math

class Initialization_spheresheet:
    def __init__(self, device='cuda'):
        self.device = device

    def create_thick_spherical_shell(self, 
        outer_radius=10.0,
        thickness=2.0,
        num_theta=20,
        num_phi=20,
        device="cuda"
    ):
        """
        生成加厚球壳的所有顶点坐标（作为整体张量返回）
    
        返回:
            vertices: 所有顶点坐标的张量 (N, 3)
        """
        # 生成角度网格
        theta = np.linspace(0, 2 * np.pi, num_theta)
        phi = np.linspace(0, np.pi, num_phi)

        # 计算内外球面顶点
        inner_radius = outer_radius - thickness
        outer_vertices = np.zeros((num_theta, num_phi, 3))
        inner_vertices = np.zeros((num_theta, num_phi, 3))

        for i in range(num_theta):
            for j in range(num_phi):
                # 外球面
                outer_vertices[i, j] = [
                outer_radius * np.sin(phi[j]) * np.cos(theta[i]),
                outer_radius * np.sin(phi[j]) * np.sin(theta[i]),
                outer_radius * np.cos(phi[j])
                ]
                # 内球面
                inner_vertices[i, j] = [
                inner_radius * np.sin(phi[j]) * np.cos(theta[i]),
                inner_radius * np.sin(phi[j]) * np.sin(theta[i]),
                inner_radius * np.cos(phi[j])
                ]

        # 转换为 GPU 张量
        outer_vertices = torch.tensor(outer_vertices, dtype=torch.float32, device=device)
        inner_vertices = torch.tensor(inner_vertices, dtype=torch.float32, device=device)

        # 将所有顶点合并为一个张量
        all_vertices = []

        for i in range(num_theta - 1):
            for j in range(num_phi - 1):
                # 外球面四边形（已经是张量）
                outer_quad = [
                    outer_vertices[i, j],
                    outer_vertices[i+1, j],
                    outer_vertices[i+1, j+1],
                    outer_vertices[i, j+1]
                ]
                # 内球面四边形（已经是张量）
                inner_quad = [
                    inner_vertices[i, j],
                    inner_vertices[i+1, j],
                    inner_vertices[i+1, j+1],
                    inner_vertices[i, j+1]
                ]
            
                # 添加四棱台的8个顶点（已经是张量）
                all_vertices.extend(outer_quad)
                all_vertices.extend(inner_quad)

        # 堆叠所有顶点并去除重复
        vertices = torch.stack(all_vertices, dim=0)
        vertices = torch.unique(vertices, dim=0)  # 去除重复顶点

        return vertices

    
