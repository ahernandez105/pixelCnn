import torch
from torch.nn import Module, LogSoftmax, NLLLoss, ReLU, LayerNorm, Conv2d, ModuleList
from torch.distributions import Categorical

class MaskedConv2dBinary(Conv2d):
    def __init__(self, m_type, in_channels, out_channels, kernel_size, padding):
        assert m_type=='A' or m_type=='B'
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.register_buffer('mask', torch.zeros_like(self.weight))

        if m_type=='A':
            self.mask[:, :, 0:kernel_size//2, :] = 1
            self.mask[:, :, kernel_size//2, 0:kernel_size//2] = 1
        else:
            self.mask[:, :, 0:kernel_size//2, :] = 1
            self.mask[:, :, kernel_size//2, 0:kernel_size//2 + 1] = 1
    
    def forward(self, x):
        self.weight.data *= self.mask
        
        return super().forward(x)

class MaskedConv2dColor(MaskedConv2dBinary):
    def __init__(self, m_type, in_channels, out_channels, kernel_size, padding):
        super().__init__(m_type, in_channels, out_channels, kernel_size, padding)
        in_idx = in_channels // 3
        out_idx = out_channels // 3

        if m_type=='A':
            # allow R channels on 2nd third filters
            self.mask[out_idx:out_idx*2, :in_idx, kernel_size//2, kernel_size//2] = 1
            # allow R and G channels on 3rd third filters
            self.mask[out_idx*2:, 0:in_idx*2, kernel_size//2, kernel_size//2 ] = 1
        else:
            # zero out the middle pixel across all input channels and filters
            self.mask[:, :, kernel_size//2, kernel_size//2] = 0
            # allow R channels on the 1st third filters
            self.mask[:out_idx, :in_idx, kernel_size//2, kernel_size//2] = 1
            # allow R and G channels on the 2nd third filters
            self.mask[out_idx:out_idx*2, 0:in_idx*2, kernel_size//2, kernel_size//2] = 1
            # allow R, G and B channels on the 3rd third filters
            self.mask[out_idx*2:, :, kernel_size//2, kernel_size//2] = 1


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
        self.net.append(ReLU())
    
    def forward(self, x):
        initial_x = x
        for module in self.net:
            x = module(x)
        
        return initial_x + x

class PixelCnn(Module):
    def __init__(
        self, in_dim, channels, kernel_size, layers, filters, 
        dist_size, masked_conv_class):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.layers = layers
        self.dist_size = dist_size
        self.mconv = masked_conv_class
        p = int((self.kernel_size - 1)/2)
    
        self.net = ModuleList()
        self.net.append(self.mconv('A', self.channels, self.filters, self.kernel_size, p))
        self.net.append(ReLU())
        for _ in range(self.layers-1):
            self.net.append(ResBlock(
                'B', self.filters, self.filters, self.kernel_size, self.mconv))
        self.net.append(self.mconv('B', self.filters, self.filters, 1, 0))
        self.net.append(ReLU())
        self.net.append(self.mconv('B', self.filters, self.dist_size*self.channels, 1, 0))
        
        self.log_softmax = LogSoftmax(dim=2)
        self.loss = NLLLoss(reduction='sum')
        print(self)
    
    def forward(self, x):
        bsize, _, _, _ = x.shape
        for module in self.net:
            x = module(x)

        return self.log_softmax(x.view(bsize, self.channels, self.dist_size, self.in_dim, self.in_dim))

    def get_loss(self, x, y):
        loss = 0
        bsize, _, _, _, _ = x.shape

        for i in range(self.channels):
            loss += self.loss(
                x[:, i, :, :, :].view(bsize, self.dist_size, self.in_dim * self.in_dim),
                y[:, i, :, :].view(bsize, self.in_dim * self.in_dim))

        return loss

    def generate_samples(self, n, dev, mean, std):
        self.eval()
        samples_in = torch.zeros((n, self.channels, self.in_dim, self.in_dim), dtype=torch.float, device=dev)
        samples_out = torch.zeros((n, self.channels, self.in_dim, self.in_dim), dtype=torch.float, device=dev)
        samples_in[:] = 1

        for row in range(self.in_dim):
            for col in range(self.in_dim):
                for channel in range(self.channels):
                    dist = Categorical(torch.exp(self(samples_in)[:, channel, :, row, col]))
                    s = dist.sample().type(torch.float)
                    samples_in[:, channel, row, col] = (s-mean[channel])/std[channel]
                    samples_out[:, channel, row, col] = s
        
        return samples_out
    
    def load(self, path, cur_dev):
        """
        Assumes model was saved on GPU. Will load based off of cur_dev.
        """
        if cur_dev == 'cpu':
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(path))
            self.to(torch.device("cuda"))