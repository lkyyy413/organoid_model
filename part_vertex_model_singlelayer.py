import torch
import numpy as np

class VertexModel:
    def __init__(self, cube_dict, k_bend, K_vol, k_membrane, growth_rates, time_step, device='cuda'):
        self.cube_dict = cube_dict  # 字典 {cube_idx: vertices_tensor (8,3)}
        self.device = device
        
        # 收集所有唯一顶点并建立索引
        self._setup_vertex_mapping()
        
        # 初始化物理参数
        self.k_bend = k_bend      # 上皮弯曲刚度
        self.K_vol = K_vol       # 体积压缩模量
        self.k_membrane = torch.full((len(self.cube_vertex_indices),), float(k_membrane), device=device)

        # 体积相关参数
        self.growth_rates = torch.full((len(self.cube_vertex_indices),), float(growth_rates), device=device)
        self.min_volumes = torch.full((len(self.cube_vertex_indices),), 0.1, device=device)   # 最小允许体积
        self.time_step = time_step

        # 新增：最大体积和表面积限制
        self.max_volumes = torch.full((len(self.cube_vertex_indices),), 2.0, device=device)  # 示例值，可根据需求调整
        self.max_areas = torch.full((len(self.cube_vertex_indices),), 10.0, device=device)   # 示例值，可根据需求调整

        self.target_volumes = torch.full((len(self.cube_vertex_indices),), 1.0, device=device)
        
        # 从目标体积推导初始目标表面积（假设为立方体）
        self.initial_volumes = self.compute_volume()  # 初始实际体积
        self.initial_areas = self.compute_surface_area()  # 初始实际表面积
        self.area_volume_ratio = self.initial_areas / self.initial_volumes**(2/3)  # 保存比例关系
        
    def _setup_vertex_mapping(self):
        """建立顶点映射关系，处理共享顶点"""
        all_vertices = []
        cube_vertex_indices = []
        
        # 用于跟踪已见过的顶点和它们的全局索引
        vertex_to_index = {}
        global_index = 0
        
        # 处理每个长方体的顶点
        for cube_idx, vertices in self.cube_dict.items():
            cube_indices = []
            
            for local_vert_idx in range(8):  # 每个长方体有8个顶点
                vertex = vertices[local_vert_idx].cpu().numpy().round(6)  # 四舍五入避免浮点误差
                vertex_tuple = tuple(vertex)
                
                if vertex_tuple not in vertex_to_index:
                    vertex_to_index[vertex_tuple] = global_index
                    all_vertices.append(vertices[local_vert_idx])
                    global_index += 1
                
                cube_indices.append(vertex_to_index[vertex_tuple])
            
            cube_vertex_indices.append(cube_indices)
        
        # 转换为张量
        self.vertices = torch.stack(all_vertices).to(self.device)  # (num_unique_vertices, 3)
        self.cube_vertex_indices = torch.tensor(cube_vertex_indices, device=self.device)  # (num_cubes, 8)
        
        # 建立反向映射：每个顶点属于哪些长方体
        self.vertex_to_cubes = [[] for _ in range(len(self.vertices))]
        for cube_idx, cube_indices in enumerate(self.cube_vertex_indices):
            for vert_idx in cube_indices:
                self.vertex_to_cubes[vert_idx].append(cube_idx)
    
    def compute_volume(self):
        """计算每个长方体的体积 (num_cubes,)"""
        # 获取每个长方体的8个顶点
        cube_vertices = self.vertices[self.cube_vertex_indices]  # (num_cubes, 8, 3)
        
        # 计算边长向量 (近似方法，假设长方体近似为矩形)
        edge1 = cube_vertices[:, 1] - cube_vertices[:, 0]  # 边长向量1
        edge2 = cube_vertices[:, 3] - cube_vertices[:, 0]  # 边长向量2
        edge3 = cube_vertices[:, 4] - cube_vertices[:, 0]  # 边长向量3
        
        # 计算体积 = |(edge1 × edge2) · edge3|
        cross = torch.cross(edge1, edge2, dim=1)
        volumes = torch.abs(torch.sum(cross * edge3, dim=1))
        
        return volumes
    
    def compute_surface_area(self):
        """优化的表面积计算"""
        cube_vertices = self.vertices[self.cube_vertex_indices]  # (num_cubes, 8, 3)
        # 预计算所有面的面积（立方体有6个面）
        face_combinations = [
            (0,1,2,3),  # 底面
            (4,5,6,7),  # 顶面
            (0,1,5,4),  # 前面
            (1,2,6,5),  # 右面
            (2,3,7,6),  # 后面
            (0,3,7,4)   # 左面
        ]
        areas = torch.zeros(len(cube_vertices), device=self.device)
        for face in face_combinations:
            # 将四边形分成两个三角形计算
            v0, v1, v2, v3 = [cube_vertices[:, i] for i in face]
            area1 = torch.norm(torch.cross(v1-v0, v2-v0, dim=1), dim=1) / 2
            area2 = torch.norm(torch.cross(v2-v0, v3-v0, dim=1), dim=1) / 2
            areas += area1 + area2
        return areas

    def update_target_volumes(self, time_step):
        """更新目标体积，并限制不超过最大体积"""
        new_volumes = self.target_volumes + time_step * self.growth_rates
        
        # 限制目标体积不超过 max_volumes
        self.target_volumes = torch.minimum(new_volumes, self.max_volumes)
        
        # 目标体积不能小于最小允许体积
        self.target_volumes = torch.maximum(self.target_volumes, self.min_volumes)
        torch.cuda.synchronize()  # 确保 CUDA 操作完成
        return self.target_volumes
    
    def get_target_areas(self):
        """计算目标表面积，并限制不超过最大表面积"""
        # 从目标体积计算理论目标表面积
        target_areas = self.area_volume_ratio * self.target_volumes**(2/3)
        
        # 限制目标表面积不超过 max_areas
        target_areas = torch.minimum(target_areas, self.max_areas)
        return target_areas
    
    def compute_volume_force(self):
        """计算体积变化导致的压力 (KV(lnV/V0 - 1))"""
        volumes = self.compute_volume()  # 当前体积
        self.target_volumes = self.update_target_volumes(self.time_step)
        pressure = self.K_vol * torch.log(volumes / self.target_volumes)
        forces = torch.zeros_like(self.vertices)
        for cube_idx in range(len(self.cube_vertex_indices)):
            vert_indices = self.cube_vertex_indices[cube_idx]
            vertices = self.vertices[vert_indices]
            center = vertices.mean(dim=0)
            # 压力方向沿径向
            force = pressure[cube_idx] * (vertices - center)
            for i, vert_idx in enumerate(vert_indices):
                forces[vert_idx] += force[i]
        return forces
    
    
    def compute_bending_force(self):
        """计算弯曲力，返回每个顶点的力张量 (num_vertices, 3)"""
        curvature = 0.1  # 示例值，实际需计算曲率
        bending_energy = self.k_bend * (2 * curvature)**2 / 2
    
        # 将能量转换为顶点力（简化示例：均匀分配到所有顶点）
        num_vertices = len(self.vertices)
        force_magnitude = bending_energy / num_vertices  # 平均分配
        forces = torch.zeros_like(self.vertices)  # (num_vertices, 3)
    
        # 实际应根据曲率梯度计算力，这里仅作示例
        forces += force_magnitude * torch.randn_like(forces)  # 添加随机方向
        return forces
    
    def compute_membrane_force(self):
        """计算膜弹性力 (k(A/A0 - 1)^2 / 2)"""
        current_areas = self.compute_surface_area()
        target_areas = self.get_target_areas()
    
        tension = self.k_membrane * (current_areas / target_areas - 1)
    
        forces = torch.zeros_like(self.vertices)
        for cube_idx in range(len(self.cube_vertex_indices)):
            vert_indices = self.cube_vertex_indices[cube_idx]
            cube_verts = self.vertices[vert_indices]  # shape: [8, 3]
        
            # 计算法向量（修正后的版本）
            # 底面 (0-1-2-3)
            vec1 = cube_verts[1] - cube_verts[0]  # shape: [3]
            vec2 = cube_verts[3] - cube_verts[0]  # shape: [3]
            normal1 = torch.cross(vec1, vec2)  # 去除dim参数
        
            # 顶面 (4-5-6-7)
            vec1 = cube_verts[5] - cube_verts[4]
            vec2 = cube_verts[7] - cube_verts[4]
            normal2 = torch.cross(vec1, vec2)
        
            # 平均法向量
            avg_normal = (normal1 + normal2) / 2.0
            avg_normal = avg_normal / (torch.norm(avg_normal) + 1e-6)
        
            # 应用力
            force = tension[cube_idx] * avg_normal
            for i, vert_idx in enumerate(vert_indices):
                forces[vert_idx] += force

        return forces

    def compute_total_force(self):
        """合并所有力，若顶点所属的任一细胞达到最大体积或表面积，则不受任何力"""
        total_force = torch.zeros_like(self.vertices)
        current_volumes = self.compute_volume()
        current_areas = self.compute_surface_area()

        # 标记达到最大限制的细胞
        max_volume_cubes = (current_volumes >= self.max_volumes)
        max_area_cubes = (current_areas >= self.max_areas)

        # 计算所有力（弯曲力、体积力、膜弹性力）
        bending_force = self.compute_bending_force()  # 形状: (num_vertices, 3)
        volume_force = self.compute_volume_force()    # 形状: (num_vertices, 3)
        membrane_force = self.compute_membrane_force()  # 形状: (num_vertices, 3)

        # 遍历所有顶点，决定是否应用力
        for vert_idx in range(len(self.vertices)):
            # 检查该顶点所属的所有细胞是否均未达到最大限制
            affected_cubes = self.vertex_to_cubes[vert_idx]  # 该顶点所属的细胞列表
            is_active = True

            for cube_idx in affected_cubes:
                if max_volume_cubes[cube_idx] or max_area_cubes[cube_idx]:
                    is_active = False  # 只要有一个细胞达到限制，顶点不受力
                    break

            if is_active:
                total_force[vert_idx] = (
                    bending_force[vert_idx] +
                    volume_force[vert_idx] +
                    membrane_force[vert_idx]
                )
            # 否则 total_force[vert_idx] 保持为零

        return total_force

    def update_vertices(self, active_cells=None):
        """更新顶点位置，可指定仅更新特定细胞（active_cells为细胞索引列表）"""
        total_forces = self.compute_total_force()
    
        if active_cells is not None:
            # 仅更新活动细胞的顶点
            active_vert_indices = set()
            for cube_idx in active_cells:
                active_vert_indices.update(self.cube_vertex_indices[cube_idx].tolist())
        
            # 仅对活动顶点应用位移
            displacement = self.time_step * total_forces
            for vert_idx in active_vert_indices:
                self.vertices[vert_idx] += displacement[vert_idx]
        else:
            # 全局更新（默认行为）
            self.vertices += self.time_step * total_forces
    
        # 更新 cube_dict
        for cube_idx, vert_indices in enumerate(self.cube_vertex_indices):
            self.cube_dict[cube_idx] = self.vertices[vert_indices]
        return self.vertices
        
    def compute_mechanical_potential(self):
        """计算顶点模型的总势能"""
        
        # 1. 体积弹性势能
        volumes = self.compute_volume()
        self.update_target_volumes(self.time_step)
        vol_potential = 0.5 * self.K_vol * torch.sum(
            (volumes - self.target_volumes)**2 / self.target_volumes
        )
        
        # 2. 膜表面势能
        areas = self.compute_surface_area()
        target_areas = self.get_target_areas()
        membrane_potential = 0.5 * torch.sum(
            self.k_membrane * (areas - target_areas)**2 / target_areas
        )
        
        # 3. 弯曲势能 (简化计算)
        curvature = 0.1  # 示例值，实际需要计算曲率
        bending_potential = 0.5 * self.k_bend * curvature**2 * torch.sum(areas)
        
        return {
            'volume': vol_potential.item(),
            'membrane': membrane_potential.item(),
            'bending': bending_potential.item(),
            'total': (vol_potential + membrane_potential + bending_potential).item()
        }