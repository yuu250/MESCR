import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttentionModule(out_channels)
        self.sa = SpatialAttentionModule()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        shortcut = self.shortcut(residual)
        x += shortcut
        x = self.relu(x)

        return x
class ResBlock(nn.Module):
    """
    Residuals Module
    @input_channel: Size of input channels
    @output_channel: Size of output channels
    """

    def __init__(self,
                 input_channel=1,
                 output_channel=4,
                 ):
        super(ResBlock, self).__init__()

        self.ResBlock = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )

    def forward(self, x):
        return self.ResBlock(x) + self.downsample(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p=1, stride=1):
        super(BasicConv, self).__init__()
        # kernel_size//2为padding 大小 图像大小保持不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p, bias=False)
        # 每次卷积后都要经过一次标准化与激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()
        self.shortcut=nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x




class MMSCNet(nn.Module):
    def __init__(self, nc=7):
        super(MMSCNet, self).__init__()

        # gri
        self.stage1_gri = nn.Sequential(
            BasicConv(3, 64, 3, stride=2),
            CBAM(64),
            BasicConv(64, 64, 3),
            BasicConv(64, 64, 3),
            BasicConv(64, 128, 3, stride=2),
            BasicConv(128, 128, 3),
            BasicConv(128, 128, 3),
        )

        # urz
        self.stage1_urz = nn.Sequential(
            BasicConv(3, 64, 3, stride=2),
            CBAM(64),
            BasicConv(64, 64, 3),
            BasicConv(64, 64, 3),
            BasicConv(64, 128, 3, stride=2),
            BasicConv(128, 128, 3),
            BasicConv(128, 128, 3),
        )

        self.stage2 = nn.Sequential(
            CBAM(256),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
            BasicConv(256, 256, 3),
            BasicConv(256, 512, 3, stride=2),
            BasicConv(512, 512, 3),
            BasicConv(512, 512, 3),
            BasicConv(512, 1024, 3, stride=2),
            BasicConv(1024, 1024, 3),
            CBAM(1024),
            BasicConv(1024, 1024, 3),
        )

        self.fc1 = torch.nn.Sequential(
            nn.Linear(16384, 1024),#16416
            torch.nn.ReLU(),
            nn.Linear(1024, 128),
            torch.nn.ReLU(),
            #nn.Dropout(p=0.3),
            # nn.Linear(512, nc),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=64,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.fc3 = torch.nn.Sequential(
            nn.Linear(64, nc)
        )

        self.mu = nn.Linear(
            in_features=64,
            out_features=1,
        )
        self.sigma = nn.Sequential(
            nn.Linear(
                in_features=64,
                out_features=1,
            ),
            nn.Softplus()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # (-1, 1, 7200)
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 4, 7200)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(          # (-1, 4, 7200)
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 16, 7200)
            nn.MaxPool1d(
                kernel_size=2,
            )                   # (-1, 16, 1800)
        )

        self.fc_x3_x4 = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.Linear(64, 32),
                )


    def forward(self, x1, x2, x3,x4):
        gri_data = x1
        urz_data = x2
        # wb_data = torch.cat([x3.float(), x4.float()], dim=1)

        gri_data = self.stage1_gri(gri_data)
        urz_data = self.stage1_urz(urz_data)

        x = torch.cat([gri_data, urz_data], axis=1)
        x = self.stage2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x3_x4 = torch.cat([x3.float(), x4.float()], dim=1)
        x3_x4_features = self.fc_x3_x4(x3_x4)
        
        x_all = torch.cat([x, x3_x4_features], axis=1)

        x_all = self.fc2(x_all)
        return self.fc3(x_all), self.mu(x_all), self.sigma(x_all)
