            
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from initialization_singlelayer import Initialization
from part_vertex_model_singlelayer import VertexModel
from analysis_singlelayer import Analysis
from part3_RD_VM_singlelayer import CoupledSimulator

def main():
    # 初始化
    init = Initialization()
    cube_dict = init.create_cells(num_cells_x=50, num_cells_y=50, num_cells_z=1, 
                                size_x=1.0, size_y=1.0, size_z=0.5, spacing=1.0)
    
    # 创建模型
    vm = VertexModel(cube_dict=cube_dict, k_bend=2*math.e-18, K_vol=0.25,
                    k_membrane=math.e-15, growth_rates=150, time_step=0.01)
    
    # 定义活动区域 (15-35行和列)
    active_region = slice(15, 35)
    # 在模拟循环中修改力学步进调用
    active_cells = [i * 50 + j for i in range(15, 35) for j in range(15, 35)]  # 活动区域的细胞索引
    
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
    for step in range(100):
        print(step)
        # 1. 反应扩散步进 (只更新活动区域)
        if step > 0:  # 跳过第一步保持初始化
            # 保存原始浓度
            #original_conc = simulator.rd.concentration.clone()
            
            # 只更新活动区域的反应扩散
            active_conc = simulator.rd.step()
            
            # 合并结果：活动区域用新值，其他区域保持原值
            simulator.rd.concentration[:, active_region, active_region] = active_conc[:, active_region, active_region]
        
        # 2. 应用化学-力学耦合（直接调用方法，但限制活动区域）
        # 临时保存原始参数
        #original_growth = vm.growth_rates.clone()
        #original_membrane = vm.k_membrane.clone()
        
        # 调用耦合方法（会打印浓度）
        simulator.apply_chemical_mechanical_coupling(threshold=13)# 修改apply_chemical_mechanical_coupling方法，使其只处理活动区域

        for i in range(15, 35):
            for j in range(15, 35):
                cell_idx = i * 50 + j  # 计算细胞索引
                concentration = simulator.get_grid_concentration(cell_idx).item()
                if concentration > 13:  # 阈值条件
                    vm.growth_rates[cell_idx] = 0.5 * concentration
                    vm.k_membrane[cell_idx] = 0.5 * concentration
        
        # 3. 力学步进 (自动只影响活动区域，因为其他区域的参数未改变)
        vm.update_vertices(active_cells=active_cells)

        # 4. 计算当前步的势能并存储
        mech_pot = vm.compute_mechanical_potential()['total']
        rd_pot = simulator.rd.compute_rd_potential()['total']
        coup_pot = simulator.compute_coupling_potential()
        
        stability_history['mechanical'].append(mech_pot)
        stability_history['rd'].append(rd_pot)
        stability_history['coupling'].append(coup_pot)
        stability_history['total'].append(mech_pot + rd_pot + coup_pot)
        
        # 每隔5步可视化
        if step % 5 == 0:
            Analysis.plot_cell_vertices(vm, save_path=f"/home/kaiyi/RD_VM/RD_VM_cuboid/3vm_rd_singlelayer/vertices_{step:04d}.pdf")
            Analysis.plot_concentration(simulator.rd.concentration[0, :], save_path=f"/home/kaiyi/RD_VM/RD_VM_cuboid/3vm_rd_singlelayer/conc_{step:04d}.pdf")
    
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
    plt.savefig("/home/kaiyi/RD_VM/RD_VM_cuboid/3vm_rd_singlelayer/potential_evolution.pdf")
    plt.close()
    
    print("\nFinal Stability Analysis:")
    print(f"Mechanical potential trend: {np.gradient(stability_history['mechanical'])[-5:]}")
    print(f"RD potential trend: {np.gradient(stability_history['rd'])[-5:]}")
    print(f"Total potential trend: {np.gradient(stability_history['total'])[-5:]}")

if __name__ == "__main__":
    main()