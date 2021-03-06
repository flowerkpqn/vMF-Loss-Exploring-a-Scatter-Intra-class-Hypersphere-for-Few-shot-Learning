import torch.nn as nn

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels,pool=True):
    if pool:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ConvNet(nn.Module):

    def __init__(self, depth=4,x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        if depth==4:
            self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, z_dim),
            )
        else:
            self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim,False),
                conv_block(hid_dim, z_dim,False),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x




