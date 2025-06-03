import torch
import torch.nn.functional as F

class RDReaction:
    def __init__(self, concentration, dx, dy, dt, device='cuda'):
        self.concentration = concentration.to(device)
        self.device = device
        if isinstance(dx, torch.Tensor):
            self.dx = dx.clone().detach().to(device)
        else:
            self.dx = torch.tensor(dx, dtype=torch.float32, device=device)
    
        if isinstance(dy, torch.Tensor):
            self.dy = dy.clone().detach().to(device)
        else:
            self.dy = torch.tensor(dy, dtype=torch.float32, device=device)
        self.dt = dt

    
    def laplacian(self, i):
        """5点离散拉普拉斯算子，支持矩形网格"""
        U_0n = torch.roll(self.concentration[i, :, :], shifts=1, dims=0)  # U_(i,j-1)
        U_n0 = torch.roll(self.concentration[i, :, :], shifts=1, dims=1)  # U_(i-1,j)
        U_00 = self.concentration[0, i, :]  # U_(i,j)
        U_p0 = torch.roll(self.concentration[i, :, :], shifts=-1, dims=1)  # U_(i+1,j)
        U_0p = torch.roll(self.concentration[i, :, :], shifts=-1, dims=0)  # U_(i,j+1)

        # x方向二阶导
        d2x = (U_p0 - 2 * U_00 + U_n0) / (self.dx**2)
        # y方向二阶导
        d2y = (U_0p - 2 * U_00 + U_0n) / (self.dy**2)
        
        L = d2x + d2y
        return L

    def reaction_term_U(self, U, V):
        """自定义反应项 for U"""
        #return (100 * U / (3.1623 + U + V) + 0.1 - U)
        return 12 - U - 4 * (U * V / (1 + U**2))

    def reaction_term_V(self, U, V):
        """自定义反应项 for V"""
        #return (100 * U / (3.1623 + U) + 3.1623 - V)
        return 0.37 * U - 0.37 * (U * V / (1 + U**2))

    def step(self, Du=1.0, Dv=1.0, dy=None):
        """时间步进，支持矩形网格"""
        diffusion_U = Du * self.laplacian(0)
        diffusion_V = Dv * self.laplacian(1)
        
        U = self.concentration[0, 0, :]
        V = self.concentration[0, 1, :]
        
        reaction_U = self.reaction_term_U(U, V)
        reaction_V = self.reaction_term_V(U, V)
        
        new_U = U + self.dt * (diffusion_U + reaction_U)
        new_V = V + self.dt * 20 * (diffusion_V + reaction_V)
        
        self.concentration = torch.stack((new_U, new_V), dim=0)
        return self.concentration
        
    def compute_rd_potential(self):
        """计算反应扩散系统的Lyapunov函数"""
        U = self.concentration[0]  # U通道 (H, W)
        V = self.concentration[1]  # V通道 (H, W)
        
        # 1. 扩散项势能 (使用实际网格间距)
        grad_U_x = (torch.roll(U, -1, dims=1) - torch.roll(U, 1, dims=1)) / (2 * self.dx)
        grad_U_y = (torch.roll(U, -1, dims=0) - torch.roll(U, 1, dims=0)) / (2 * self.dy)
    
        grad_V_x = (torch.roll(V, -1, dims=1) - torch.roll(V, 1, dims=1)) / (2 * self.dx)
        grad_V_y = (torch.roll(V, -1, dims=0) - torch.roll(V, 1, dims=0)) / (2 * self.dy)
        
        diffusion_potential = 0.5 * torch.sum(
            grad_U_x**2 + grad_U_y**2 + 
            grad_V_x**2 + grad_V_y**2
        ) * torch.mean(self.dx * self.dy)
        
        # 2. 反应项势能 (根据具体反应方程)
        reaction_potential = torch.sum(
            12 * U - 0.5 * U**2 - 2 * V * torch.log(1 + U**2) + 
            0.185 * U**2 - 0.185 * U * V**2 / (1 + U**2)) * torch.mean(self.dx * self.dy)
        
        
        return {
            'diffusion': diffusion_potential.item(),
            'reaction': reaction_potential.item(),
            'total': (diffusion_potential + reaction_potential).item()
        }