import torch
from torch.nn import ReLU, ModuleList, Module

class ResBlock(Module):
    def __init__(self, m_type, in_channels, out_channels, kernel_size, masked_conv_class):
        super().__init__()
        self.net = ModuleList()
        self.net.append(masked_conv_class(m_type, in_channels, out_channels, 1, 0))
        self.net.append(ReLU())
        p = int((kernel_size - 1)/2)
        self.net.append(masked_conv_class(m_type, in_channels, out_channels, kernel_size, p))
        self.net.append(ReLU())
        self.net.append(masked_conv_class(m_type, in_channels, out_channels, 1, 0))
        self.net.append(LayerNormPixel(out_channels))
        self.net.append(ReLU())
    
    def forward(self, x):
        initial_x = x
        for module in self.net:
            x = module(x)
        
        return initial_x + x
