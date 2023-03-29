import torch

class Nesterov(torch.optim.Optimizer):
    def __init__(self, params, lr = 0.1, alpha_0 = 0, alpha_1 = 1, weight = 1):
        old_params = [param.detach().clone() for param in params]
        defaults = dict(lr = lr, alpha_0 = alpha_0, alpha_1 = alpha_1, old_params = old_params, weight = weight)
        super(Nesterov, self).__init__(params, defaults)
    
    def __setstate__(self, state):
            return super(Nesterov, self).__setstate__(state)
        
    def step(self, closure = None):
    
            if closure is None:
                raise ValueError("Closure required for recompute of the loss")
            
            for group in self.param_groups:
    
                alpha_0 = group['alpha_0']
                alpha_1 = group['alpha_1']
                lr = group['lr']
                weight = group['weight']
    
                for B_t1, B_t0 in zip(list(group['params']), group['old_params']):
    
                    # Linear extrapolation
    
                    Bt1_data = B_t1.detach().clone()
                    S_t1 = Bt1_data + ((alpha_0 - 1) / alpha_1) * (Bt1_data - B_t0)
    
                    S_t1.require_grad_(True)
    
                    # Compute the loss for the extrapolation point
    
                    loss = closure(S_t1)
                    d_St1 = S_t1.grad
    
                    while True:
                        
                        # Line search                        
                        A_temp = S_t1 - d_St1 * lr
                        # Compute the singular value decomposition
                        U, a, Vh = torch.linalg.svd(A_temp)
                        b = torch.abs(a - lr * weight) # Not sure about this
                        B_temp = torch.mathmul(torch.matmul(U, torch.diag(b)), Vh)
                        # Update the learning rate:
                        gt1 = loss + torch.sum(d_St1 *  (B_temp - S_t1)) + \
                                   torch.sum((B_temp - S_t1) ** 2) / (2.0 * lr)
                        lr /= 2
                        if gt1 >= closure(B_temp):
                            break
                    if closure(Bt1_data) >= closure(B_temp):
                        B_t1.data = B_temp
                group['alpha_0'] = group['alpha_1']
                group['alpha_1'] = (torch.sqrt((alpha_1 * 2) ** 2 + 1) + 1) / 2
    