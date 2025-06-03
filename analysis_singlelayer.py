import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

class Analysis:
    @staticmethod    
    def plot_cell_vertices(vertex_model, save_path=None):
        """绘制完整的长方体（顶点+边），并保持xyz轴等比例
    
        参数:
            vertex_model: VertexModel实例，包含vertices和cube_vertex_indices
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # 定义长方体的12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ]
    
        # 获取顶点和立方体索引
        vertices = vertex_model.vertices.cpu().numpy()
        cube_vertex_indices = vertex_model.cube_vertex_indices.cpu().numpy()
    
        # 绘制每个长方体
        for cube_indices in cube_vertex_indices:
            cube_vertices = vertices[cube_indices]  # 获取当前立方体的8个顶点
            for edge in edges:
                ax.plot(
                    [cube_vertices[edge[0], 0], cube_vertices[edge[1], 0]],
                    [cube_vertices[edge[0], 1], cube_vertices[edge[1], 1]],
                    [cube_vertices[edge[0], 2], cube_vertices[edge[1], 2]],
                    color='r', linewidth=1
                )
    
        # 获取所有顶点的坐标范围
        min_val = np.min(vertices)
        max_val = np.max(vertices)
    
        # 设置xyz轴相同的范围
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Cell Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免内存泄漏
        else:
            plt.show()

    @staticmethod
    def plot_hex_prisms(vertex_model, save_path=None):
        """绘制六棱柱阵列（仅边线），保持xyz轴等比例
    
        参数:
            vertex_model: VertexModel实例，包含vertices和cube_vertex_indices
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 定义六棱柱的18条边（底面6条+顶面6条+侧面6条）
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],  # 底面六边形
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6],  # 顶面六边形
            [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]  # 侧面连接边
        ]

        # 获取顶点和六棱柱索引（确保转移到CPU）
        vertices = vertex_model.vertices.cpu().numpy()
        hex_vertex_indices = vertex_model.cube_vertex_indices.cpu().numpy()

        # 绘制每个六棱柱的边
        for hex_indices in hex_vertex_indices:
            hex_vertices = vertices[hex_indices]  # 获取当前六棱柱的12个顶点
            for edge in edges:
                ax.plot(
                    [hex_vertices[edge[0], 0], hex_vertices[edge[1], 0]],
                    [hex_vertices[edge[0], 1], hex_vertices[edge[1], 1]],
                    [hex_vertices[edge[0], 2], hex_vertices[edge[1], 2]],
                    color='r', linewidth=1  # 保持红色线框
                )

        # 设置等比例坐标轴
        min_val = np.min(vertices)
        max_val = np.max(vertices)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Hexagonal Prism Visualization')
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_concentration(concentration, save_path=None):
        """绘制浓度场"""
        plt.figure(figsize=(8, 6))
        plt.imshow(concentration.cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title('Concentration Field')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免内存泄漏
        else:
            plt.show()
        
    @staticmethod
    def save_vertices(vertices, filename):
        """
        保存顶点数据为VTK文件
        
        参数:
            vertices: 顶点张量 (N, 3)
            filename: 输出文件名 (如 "step_10.vtk")
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 转换为numpy数组
        points = vertices.cpu().numpy()
        
        # 创建点云
        cloud = pv.PolyData(points)
        
        # 保存为VTK文件
        cloud.save(filename)
        print(f"Vertices data saved to {filename}")

    @staticmethod
    def save_concentration(concentration, filename):
        """
        保存浓度场为NumPy二进制文件
        
        参数:
            concentration: 浓度张量 (B, C, H, W) 或 (H, W)
            filename: 输出文件名 (如 "conc_10.npy")
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 转换为numpy数组
        if concentration.dim() == 3:  # 如果是批次数据，取第一个
            conc_np = concentration[0, 0].cpu().numpy()  # 取第一个batch和通道
        elif concentration.dim() == 2:  # 如果是单张图
            conc_np = concentration.cpu().numpy()
        else:
            raise ValueError("unsupported concentration tensor dimension")
        
        # 保存为npy文件
        np.save(filename, conc_np)
        print(f"Concentration data saved to {filename}")

    @staticmethod
    def save_simulation_state(vertex_model, concentration, step, output_dir="output"):
        """
        一站式保存当前模拟状态
        
        参数:
            vertex_model: VertexModel实例
            concentration: 浓度张量
            step: 当前步数
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存顶点
        vtk_path = os.path.join(output_dir, f"vertices_step_{step:04d}.vtk")
        Analysis.save_vertices(vertex_model.vertices, vtk_path)
        
        # 保存浓度
        npy_path = os.path.join(output_dir, f"concentration_step_{step:04d}.npy")
        Analysis.save_concentration(concentration, npy_path)
        
    @staticmethod
    def plot_potential(stability_results, save_path=None):
        """
        绘制势能变化曲线
        
        参数:
            stability_results: 来自CoupledSimulator.analyze_stability()的结果
            save_path: 可选，图片保存路径
        """
        plt.figure(figsize=(12, 4))
        
        # 1. 提取数据（兼容新旧格式）
        mech_values = stability_results['mechanical']['values']
        rd_values = stability_results['rd']['values']
        coup_values = stability_results['coupling']['values']
        total_values = stability_results['total']['values']
    
        # 2. 力学势能
        plt.subplot(131)
        plt.plot(mech_values, 'r-', label='Total')
        plt.title("Mechanical Potential")
        plt.xlabel("Time step")
        plt.ylabel("Energy")
        plt.legend()
    

        # 3. 反应扩散势能
        plt.subplot(132)
        plt.plot(rd_values, 'b-', label='Total')
        plt.title("RD Potential")
        plt.xlabel("Time step")
        plt.ylabel("Energy")
        plt.legend()
    
        # 4. 总势能
        plt.subplot(133)
        plt.plot(total_values, 'k-', label='Total')
        plt.plot(coup_values, 'm--', label='Coupling')
        plt.title("Combined Potential")
        plt.xlabel("Time step")
        plt.ylabel("Energy")
        plt.legend()
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()