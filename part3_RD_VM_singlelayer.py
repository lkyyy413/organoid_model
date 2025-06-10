import torch
import numpy as np
import math
from part3_rd_reaction_singlelayer import RDReaction

class CoupledSimulator:
    def __init__(self, vertex_model, rd_model=None, dt=0.01, coupling_strength=0.1):
        """
        优化版耦合模拟器
        - 保持2D反应扩散网格与细胞的一一对应
        - 每个细胞的力学参数受对应网格点浓度控制
        """
        self.vm = vertex_model
        self.coupling = coupling_strength
        
        # 验证细胞数与浓度网格匹配
        num_cells = len(vertex_model.cube_vertex_indices)
        if rd_model is None:
            # 自动创建匹配的浓度场 (1 batch, 2 channels, height, width)
            grid_size = int(num_cells**0.5)  # 假设是正方形网格
            initial_concentration = self.init_concentration(grid_shape=(1, 2, grid_size, grid_size), low=0, high=1)[0]
            self.rd = RDReaction(initial_concentration, dx=torch.ones(grid_size, grid_size), dy=torch.ones(grid_size, grid_size), dt=dt)
        else:
            self.rd = rd_model
            expected_cells = self.rd.concentration.shape[1] * self.rd.concentration.shape[2]
            if num_cells != expected_cells:
                raise ValueError(f"细胞数{num_cells}与浓度网格{expected_cells}不匹配")
        
        # 建立细胞到网格的索引映射 (假设细胞按行优先排列)
        self.cell_to_grid = self._create_mapping(num_cells)
        # 预计算网格间距
        #self.update_grid_spacing()

    def init_concentration(self, grid_shape=(100, 100), low=0, high=1):
        """初始化浓度场（自定义范围）"""
        return torch.rand(*grid_shape) * (high - low) + low
    
    def update_grid_spacing(self):
        """根据当前细胞位置更新网格间距"""
        grid_size = int(len(self.vm.cube_vertex_indices)**0.5)
        dx_grid = torch.ones(grid_size, grid_size)
        dy_grid = torch.ones(grid_size, grid_size)
        
        # 计算每个细胞中心位置
        cell_centers = []
        for indices in self.vm.cube_vertex_indices:
            vertices = self.vm.vertices[indices]
            center = vertices.mean(dim=0)
            cell_centers.append(center)
        cell_centers = torch.stack(cell_centers)
        
        # 计算x和y方向的间距
        for i in range(grid_size):
            for j in range(grid_size):
                grid_idx = i * grid_size + j
                # x方向间距(与右侧细胞)
                if j < grid_size - 1:
                    right_idx = i * grid_size + (j + 1)
                    dx_grid[i, j] = (cell_centers[right_idx] - cell_centers[grid_idx]).norm()
                # y方向间距(与下方细胞)
                if i < grid_size - 1:
                    bottom_idx = (i + 1) * grid_size + j
                    dy_grid[i, j] = (cell_centers[bottom_idx] - cell_centers[grid_idx]).norm()
        
        # 更新RD模型中的间距
        self.rd.dx = dx_grid.to(self.vm.vertices.device)
        self.rd.dy = dy_grid.to(self.vm.vertices.device)
    
    def _create_mapping(self, num_cells):
        """创建细胞索引到网格坐标的映射"""
        grid_w = int(num_cells**0.5)
        return [(i//grid_w, i%grid_w) for i in range(num_cells)]

    def get_grid_concentration(self, cell_idx):
        """获取对应网格点的浓度"""
        i, j = self.cell_to_grid[cell_idx]
        return self.rd.concentration[1, i, j]  # 取batch 0的所有通道
    
    def apply_chemical_mechanical_coupling(self, threshold=15):
        """化学信号对力学参数的影响(仅在浓度超过阈值时生效)"""
        for cell_idx in range(len(self.vm.cube_vertex_indices)):
            # 获取该细胞对应的浓度 (U分量)
            concentration = self.get_grid_concentration(cell_idx).item()  # 取通道0(U)
        
            # 只有当浓度超过阈值时才影响力学参数
            if concentration > threshold:
                self.vm.k_membrane[cell_idx] = math.e+15
    
    def compute_coupling_potential(self):
        """计算化学-力学耦合势能"""
        coupling_potential = 0.0
        for cell_idx in range(len(self.vm.cube_vertex_indices)):
            i, j = self.cell_to_grid[cell_idx]
            U = self.rd.concentration[0, i, j]
            
            # 耦合势能与浓度和力学参数的乘积相关
            coupling_potential += self.coupling * U * (
                self.vm.growth_rates[cell_idx] + 
                self.vm.k_membrane[cell_idx]
            )
        return coupling_potential.item()
    
    def analyze_stability(self, window_size=5):
        """执行稳定性分析"""
        results = {
            'mechanical': [],
            'rd': [],
            'coupling': [],
            'total': []
        }
        
        for _ in range(window_size):
            # 计算各子系统势能
            mech_pot = self.vm.compute_mechanical_potential()
            rd_pot = self.rd.compute_rd_potential()
            coup_pot = self.compute_coupling_potential()
            
            # 记录结果
            results['mechanical'].append(mech_pot)
            results['rd'].append(rd_pot)
            results['coupling'].append(coup_pot)
            results['total'].append(mech_pot['total'] + rd_pot['total'] + coup_pot)
            
            # 前进一步模拟
            self.step()
        
        # 计算稳定性指标
        stability = {}
        for key in ['mechanical', 'rd', 'coupling', 'total']:
            values = [x['total'] if isinstance(x, dict) else x for x in results[key]]
            gradients = np.gradient(values)
            stability[key] = {
                'values': values,
                'gradients': gradients,
                'is_stable': all(abs(g) < 1e-3 for g in gradients)
            }
        
        return stability

    def step(self):
        """耦合的时间步进"""
        # 1. 更新网格间距
        #self.update_grid_spacing()
        
        # 2. 反应扩散步进
        self.rd.step()
        
        # 3. 应用化学-力学耦合
        self.apply_chemical_mechanical_coupling()
        
        # 4. 力学步进
        self.vm.update_vertices()
        
        return self.vm.vertices, self.rd.concentration