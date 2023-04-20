import math 
import torch
import torch.nn as nn


class MVl(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.flatten = nn.Flatten()
        self.layers = nn.Linear(input_shape[0] * input_shape[1], output_shape[0] * output_shape[1])
    
    def forward(self, x):
        x = self.flatten(x)
        output = self.layers(x)
        return output.view((output.shape[0], self.output_shape[0], self.output_shape[1]))




class RRMVL(nn.Module):
    def __init__(self, rank, input_shape, output_shape):
        super(RRMVL, self).__init__()
        self.output_shape = output_shape
        self.rank = rank
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], self.rank),
            nn.Linear(self.rank, output_shape[0] * output_shape[1])
        )
    
    def forward(self, x):
        x = self.flatten(x)
        output = self.layers(x)
        return output.view((output.shape[0], self.output_shape[0], self.output_shape[1]))
    
class TraceLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight = nn.Parameter(torch.empty(self.output_shape[0], self.output_shape[1], 
                                               self.input_shape[0], self.input_shape[1]))
        self.bias = nn.Parameter(torch.empty(self.output_shape))
        self.reset_parameters()
    def reset_parameters(self) -> None:

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        x = torch.matmul(torch.transpose(self.weight, 2, 3).unsqueeze(2), x)
        x = x.permute(2,0,1,3,4)
        x = torch.diagonal(x, dim1=3, dim2= 4)
        x = x.sum(dim = -1) + self.bias
        return x
    

class KrausLayer(nn.Module):

    def __init__(self, input_shape, output_dim, rank):
        super(KrausLayer, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        self.weight = nn.Parameter(torch.empty((self.rank, self.output_dim, input_shape[-1])))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        tmp = torch.matmul(self.weight[0], x)
        tmp = tmp.permute(1,0,2)
        Y0 = torch.matmul(tmp, self.weight[0].transpose(0, 1))
        acc = [Y0]
        for r in range(1, self.rank):
            tmp = torch.matmul(self.weight[r], x)
            tmp = tmp.permute(1,0,2)
            Yr = torch.matmul(tmp, self.weight[r].transpose(0, 1))
            acc.append(Yr)
        Y = torch.sum(torch.stack(acc), dim=0).permute(1,0,2)
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.output_dim)