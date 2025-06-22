import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F


class DISTS(torch.nn.Module):
    def __init__(self, device, as_loss=True, in_channels=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dists_metric = pyiqa.create_metric('dists', device=device, as_loss=as_loss)
        self.sobel_x = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float, requires_grad=False, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float, requires_grad=False,device=device).view(1, 1, 3, 3)
        
    def sobel_operator(self, img_tensor, in_channels=3):
        '''img_tensor:[B.C.H.W]'''
        x = F.conv2d(img_tensor, self.sobel_x.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
        y = F.conv2d(img_tensor, self.sobel_y.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
        
        out = torch.sqrt(x**2+y**2+ 1e-6)
        
        return  out
    def forward(self, img_tensor_1:torch.Tensor, img_tensor_2:torch.Tensor):
        dists_loss = self.dists_metric(img_tensor_1, img_tensor_2).mean()
        sobel_dists_loss = self.dists_metric(self.sobel_operator(img_tensor_1), self.sobel_operator(img_tensor_2)).mean()
        return (dists_loss + sobel_dists_loss)/2