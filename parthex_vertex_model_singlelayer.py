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
        self.k_membrane = torch.full((len(self.cube_vertex_indices),), float(k_membrane), device=device)  # 膜弹性刚度

        # 体积相关参数
        self.growth_rates = torch.full((len(self.cube_vertex_indices),), float(growth_rates), device=device)  # 生长率
        self.min_volumes = torch.full((len(self.cube_vertex_indices),), 0.1, device=device)   # 最小允许体积
        self.time_step = time_step

        # 新增：最大体积和表面积限制
        self.max_volumes = torch.full((len(self.cube_vertex_indices),), 5.0, device=device)  # 示例值，可根据需求调整
        self.max_areas = torch.full((len(self.cube_vertex_indices),), 15.0, device=device)   # 示例值，可根据需求调整

        self.target_volumes = torch.full((len(self.cube_vertex_indices),), 1.0, device=device)
        
        # 从目标体积推导初始目标表面积（假设为立方体）
        self.initial_volumes = self.compute_volume()  # 初始实际体积
        self.initial_areas = self.compute_surface_area()  # 初始实际表面积
        self.area_volume_ratio = self.initial_areas / self.initial_volumes**(2/3)  # 保存比例关系
        
    def _setup_vertex_mapping(self):
        """建立顶点映射关系，处理共享顶点（适配12顶点六棱柱）"""
        all_vertices = []
        cube_vertex_indices = []
        vertex_to_index = {}
        global_index = 0

        for cube_idx, vertices in self.cube_dict.items():
            cube_indices = []
            
            for local_vert_idx in range(12):  # 六棱柱有12个顶点
                vertex = vertices[local_vert_idx].cpu().numpy().round(6)
                vertex_tuple = tuple(vertex)
                
                if vertex_tuple not in vertex_to_index:
                    vertex_to_index[vertex_tuple] = global_index
                    all_vertices.append(vertices[local_vert_idx])
                    global_index += 1
                
                cube_indices.append(vertex_to_index[vertex_tuple])
            
            cube_vertex_indices.append(cube_indices)
        
        self.vertices = torch.stack(all_vertices).to(self.device)  # (num_unique_vertices, 3)
        self.cube_vertex_indices = torch.tensor(cube_vertex_indices, device=self.device)  # (num_cubes, 12)
        
        # 建立反向映射
        self.vertex_to_cubes = [[] for _ in range(len(self.vertices))]
        for cube_idx, cube_indices in enumerate(self.cube_vertex_indices):
            for vert_idx in cube_indices:
                self.vertex_to_cubes[vert_idx].append(cube_idx)

    def compute_volume(self):
        """计算六棱柱体积（通过底面面积×高度）"""
        cube_vertices = self.vertices[self.cube_vertex_indices]  # (num_cubes, 12, 3)
        
        # 计算底面面积（六边形）
        bottom_verts = cube_vertices[:, :6]  # 底面6个顶点
        center = bottom_verts.mean(dim=1)
        vecs = bottom_verts - center.unsqueeze(1)
        area = 0.5 * torch.sum(
            torch.norm(torch.cross(vecs[:, :-1], vecs[:, 1:]), dim=2),
            dim=1
        )
        
        # 计算高度（取底面到顶面中心的距离）
        top_center = cube_vertices[:, 6:].mean(dim=1)
        height = torch.norm(top_center - center, dim=1)
        
        return area * height

    def compute_surface_area(self):
        cube_vertices = self.vertices[self.cube_vertex_indices]  # (num_cubes, 12, 3)

        def hexagon_area(verts):
            center = verts.mean(dim=1, keepdim=True)
            vecs = verts - center
            cross = torch.cross(vecs[:, :-1], vecs[:, 1:], dim=2)
            return 0.5 * torch.norm(cross, dim=2).sum(dim=1)

        # 底面和顶面面积
        bottom_area = hexagon_area(cube_vertices[:, :6])
        top_area = hexagon_area(cube_vertices[:, 6:12])

        # 侧面面积（6个矩形，拆分为三角形）
        side_areas = []
        for i in range(6):
            v0 = cube_vertices[:, i]
            v1 = cube_vertices[:, (i + 1) % 6]
            v2 = cube_vertices[:, i + 6]
            v3 = cube_vertices[:, (i + 1) % 6 + 6]
        
            # 矩形拆分为两个三角形
            area1 = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
            area2 = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1, dim=1), dim=1)
            side_areas.append(area1 + area2)

        total_side_area = torch.stack(side_areas).sum(dim=0)
        total_area = bottom_area + top_area + total_side_area
        return total_area

    # 以下方法与原版保持一致（仅内部计算使用新的体积/面积方法）
    def update_target_volumes(self, time_step):
        new_volumes = self.target_volumes + time_step * self.growth_rates
        self.target_volumes = torch.minimum(new_volumes, self.max_volumes)
        self.target_volumes = torch.maximum(self.target_volumes, self.min_volumes)
        return self.target_volumes

    def get_target_areas(self):
        target_areas = self.area_volume_ratio * self.target_volumes**(2/3)
        target_areas = torch.minimum(target_areas, self.max_areas)
        return target_areas

    def compute_bending_force(self):
        #curvature = 0.1
        curvature = 10
        bending_energy = self.k_bend * (2 * curvature)**2 / 2
        num_vertices = len(self.vertices)
        force_magnitude = bending_energy / num_vertices
        forces = torch.zeros_like(self.vertices)
        forces += force_magnitude * torch.randn_like(forces)
        return forces

    def compute_volume_force(self):
        volumes = self.compute_volume()
        self.target_volumes = self.update_target_volumes(self.time_step)
        pressure = self.K_vol * torch.log(volumes / self.target_volumes)
        
        forces = torch.zeros_like(self.vertices)
        for cube_idx in range(len(self.cube_vertex_indices)):
            vert_indices = self.cube_vertex_indices[cube_idx]
            vertices = self.vertices[vert_indices]
            center = vertices.mean(dim=0)
            force = pressure[cube_idx] * (vertices - center)
            for i, vert_idx in enumerate(vert_indices):
                forces[vert_idx] += force[i]
        return forces

    def compute_membrane_force(self):
        current_areas = self.compute_surface_area()
        target_areas = self.get_target_areas()
        tension = self.k_membrane * (current_areas / target_areas - 1)
        
        forces = torch.zeros_like(self.vertices)
        for cube_idx in range(len(self.cube_vertex_indices)):
            vert_indices = self.cube_vertex_indices[cube_idx]
            cube_verts = self.vertices[vert_indices]
            
            # 计算平均法向量（底面和顶面法向的平均）
            vec1 = cube_verts[1] - cube_verts[0]
            vec2 = cube_verts[2] - cube_verts[0]
            normal1 = torch.cross(vec1, vec2)
            
            vec1 = cube_verts[7] - cube_verts[6]
            vec2 = cube_verts[8] - cube_verts[6]
            normal2 = torch.cross(vec1, vec2)
            
            avg_normal = (normal1 + normal2) / 2.0
            avg_normal = avg_normal / (torch.norm(avg_normal) + 1e-6)
            
            force = tension[cube_idx] * avg_normal
            for i, vert_idx in enumerate(vert_indices):
                forces[vert_idx] += force
        return forces

    def compute_total_force(self):
        total_force = torch.zeros_like(self.vertices)
        current_volumes = self.compute_volume()
        current_areas = self.compute_surface_area()
        max_volume_cubes = (current_volumes >= self.max_volumes)
        max_area_cubes = (current_areas >= self.max_areas)

        bending_force = self.compute_bending_force()
        volume_force = self.compute_volume_force()
        membrane_force = self.compute_membrane_force()

        for vert_idx in range(len(self.vertices)):
            affected_cubes = self.vertex_to_cubes[vert_idx]
            is_active = True
            for cube_idx in affected_cubes:
                if max_volume_cubes[cube_idx] or max_area_cubes[cube_idx]:
                    is_active = False
                    break
            if is_active:
                total_force[vert_idx] = bending_force[vert_idx] + volume_force[vert_idx] + membrane_force[vert_idx]
        return total_force

    def update_vertices(self, active_cells=None):
        total_forces = self.compute_total_force()
        if active_cells is not None:
            active_vert_indices = set()
            for cube_idx in active_cells:
                active_vert_indices.update(self.cube_vertex_indices[cube_idx].tolist())
            displacement = self.time_step * total_forces
            for vert_idx in active_vert_indices:
                self.vertices[vert_idx] += displacement[vert_idx]
        else:
            self.vertices += self.time_step * total_forces
        
        # 更新 cube_dict
        for cube_idx, vert_indices in enumerate(self.cube_vertex_indices):
            self.cube_dict[cube_idx] = self.vertices[vert_indices]
        return self.vertices

    def compute_mechanical_potential(self):
        volumes = self.compute_volume()
        self.update_target_volumes(self.time_step)
        vol_potential = 0.5 * self.K_vol * torch.sum(
            (volumes - self.target_volumes)**2 / self.target_volumes
        )
        
        areas = self.compute_surface_area()
        target_areas = self.get_target_areas()
        membrane_potential = 0.5 * torch.sum(
            self.k_membrane * (areas - target_areas)**2 / target_areas
        )
        
        curvature = 0.1
        bending_potential = 0.5 * self.k_bend * curvature**2 * torch.sum(areas)
        
        return {
            'volume': vol_potential.item(),
            'membrane': membrane_potential.item(),
            'bending': bending_potential.item(),
            'total': (vol_potential + membrane_potential + bending_potential).item()
        }