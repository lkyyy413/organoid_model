            
import torch
#print(torch.cuda.is_available())
import math
import numpy as np
import matplotlib.pyplot as plt
from initialization_singlelayer import Initialization
from parthex_vertex_model_singlelayer import VertexModel
from analysis_singlelayer import Analysis

def main(x_min=None, x_max=None, y_min=None, y_max=None):
    # 初始化
    num_cells_x, num_cells_y = 20, 20
    init = Initialization()
    cube_dict = init.create_cells_hex(num_cells_x=num_cells_x, num_cells_y=num_cells_y, 
                                     num_cells_z=1, radius=1.0, height=0.5)
    #grid_shape = (50, 50)  # 与细胞网格一致

    # 创建模型
    vm = VertexModel(cube_dict=cube_dict, k_bend=2*math.e-18, K_vol=0.25, 
                    k_membrane=math.e-15, growth_rates=15, time_step=0.01)

    # 动态确定活动单元
    if all(v is not None for v in [x_min, x_max, y_min, y_max]):
        active_cubes = {
            cube_idx for cube_idx in cube_dict.keys()
            if (x_min <= (cube_idx % num_cells_x) < x_max) and 
               (y_min <= (cube_idx // num_cells_x) < y_max)
        }
    else:
        active_cubes = set(cube_dict.keys())  # 未指定范围时激活全部单元
    #print("Number of active cubes:", len(active_cubes))

    # 存储势能数据
    potential_data = {
        'volume': [],
        'membrane': [],
        'bending': [],
        'total': []
    }
    
    # 预热
    for step in range(50):
        print("step:", step)
        # 保存原始顶点位置
        #original_vertices = vm.vertices.clone()
        
        # 更新顶点位置（仅对active_cubes施加力）
        total_forces = vm.compute_total_force()
        
        # 仅对active_cubes中的顶点施加力
        for cube_idx in active_cubes:
            vert_indices = vm.cube_vertex_indices[cube_idx]
            vm.vertices[vert_indices] += vm.time_step * total_forces[vert_indices]
        
        # 更新cube_dict
        for cube_idx, vert_indices in enumerate(vm.cube_vertex_indices):
            vm.cube_dict[cube_idx] = vm.vertices[vert_indices]
            
        # 计算势能并存储
        potentials = vm.compute_mechanical_potential()
        for key in potential_data:
            potential_data[key].append(potentials[key])
        
        if step % 5 == 0:
            # 可视化
            #Analysis.plot_cell_vertices(vm, save_path=f"/home/kaiyi/RD_VM/1onlyvmhex_vertices_singlelayer/vertices_{step:04d}.pdf")
            Analysis.plot_hex_prisms(vm, save_path=f"/home/kaiyi/RD_VM/RD_VM_hexagon/1onlyvmhex_vertices_singlelayer/vertices_{step:04d}.pdf")
            
    # 绘制势能变化图
    plt.figure(figsize=(10, 6))
    for key in ['volume', 'membrane', 'bending', 'total']:
        plt.plot(potential_data[key], label=key)
    plt.xlabel('Time Step')
    plt.ylabel('Potential Energy')
    plt.title('Potential Energy Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/kaiyi/RD_VM/RD_VM_hexagon/1onlyvmhex_vertices_singlelayer/potential_energy_evolution.pdf")
    plt.close()

if __name__ == "__main__":
    #main(x_min=15, x_max=35, y_min=15, y_max=35)
    main()
    
    

    