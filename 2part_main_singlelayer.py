            
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from initialization_singlelayer import Initialization
from parthex_vertex_model_singlelayer import VertexModel
from analysis_singlelayer import Analysis
from part2_RD_VM_singlelayer import CoupledSimulator

def main(x_min=None, x_max=None, y_min=None, y_max=None, threshold=13):
    # 初始化
    num_cells_x, num_cells_y = 20, 20
    init = Initialization()
    cube_dict = init.create_cells_hex(num_cells_x=num_cells_x, num_cells_y=num_cells_y, 
                                     num_cells_z=1, radius=1.0, height=0.5)
    
    # 创建模型
    vm = VertexModel(cube_dict=cube_dict, k_bend=2*math.e-18, K_vol=0.25,
                    k_membrane=math.e-15, growth_rates=15, time_step=0.01)
    
    # 确定活动区域
    if all(v is not None for v in [x_min, x_max, y_min, y_max]):
        # 使用指定区域
        active_cells = [
            i * num_cells_x + j 
            for i in range(y_min, y_max) 
            for j in range(x_min, x_max)
        ]
        active_region = (slice(y_min, y_max), slice(x_min, x_max))
    else:
        # 默认全部区域激活
        active_cells = list(range(num_cells_x * num_cells_y))
        active_region = (slice(None), slice(None))
    
    # 创建耦合模拟器
    simulator = CoupledSimulator(vm, coupling_strength=0.5, dt=0.01)

    # 初始化势能记录
    stability_history = {
        'mechanical': [],
        'rd': [],
        'coupling': [],
        'total': []
    }
    
    # 模拟循环
    for step in range(1000):
        print(f"Step {step}")
        
        # 1. 反应扩散步进 (跳过第一步保持初始化)
        if step > 0:
            active_conc = simulator.rd.step()
            # 仅更新活动区域
            simulator.rd.concentration[:, active_region[0], active_region[1]] = \
                active_conc[:, active_region[0], active_region[1]]
        
        # 2. 化学-力学耦合 (仅活动区域)
        for cell_idx in active_cells:
            concentration = simulator.get_grid_concentration(cell_idx).item()
            if concentration > threshold:
                vm.growth_rates[cell_idx] = 0.5 * concentration
                vm.k_membrane[cell_idx] = 0.5 * concentration
        
        # 3. 力学步进 (仅活动区域)
        vm.update_vertices(active_cells=active_cells)
        print("Vertices range:", vm.vertices.min(), vm.vertices.max())

        # 4. 计算当前步的势能并存储
        mech_pot = vm.compute_mechanical_potential()['total']
        rd_pot = simulator.rd.compute_rd_potential()['total']
        coup_pot = simulator.compute_coupling_potential()
        
        stability_history['mechanical'].append(mech_pot)
        stability_history['rd'].append(rd_pot)
        stability_history['coupling'].append(coup_pot)
        stability_history['total'].append(mech_pot + rd_pot + coup_pot)
        
        # 5. 可视化
        if step % 100 == 0:
            Analysis.plot_hex_prisms(vm, 
                save_path=f"/home/kaiyi/RD_VM/RD_VM_hexagon/2vm_rd_singlelayer/vertices_{step:04d}.pdf")
            Analysis.plot_concentration(simulator.rd.concentration[0], 
                save_path=f"/home/kaiyi/RD_VM/RD_VM_hexagon/2vm_rd_singlelayer/conc_{step:04d}.pdf")
    
    #torch.save(vm.vertices, "/home/kaiyi/RD_VM/RD_VM_hexagon/2vm_rd_singlelayer/vertices_final_1000.pt")
    # 绘制势能变化曲线
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(stability_history['total']))
    
    for key in ['mechanical', 'rd', 'coupling', 'total']:
        plt.plot(steps, stability_history[key], label=key, alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Potential Energy')
    plt.title('Potential Energy Evolution Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/kaiyi/RD_VM/RD_VM_hexagon/2vm_rd_singlelayer/potential_evolution.pdf")
    plt.close()
    
    print("\nFinal Stability Analysis:")
    print(f"Mechanical potential trend: {np.gradient(stability_history['mechanical'])[-5:]}")
    print(f"RD potential trend: {np.gradient(stability_history['rd'])[-5:]}")
    print(f"Total potential trend: {np.gradient(stability_history['total'])[-5:]}")

if __name__ == "__main__":
    main()