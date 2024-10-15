import torch.nn as nn
import torch

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


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p=1, stride=1):
        super(BasicConv, self).__init__()
        # kernel_size//2为padding 大小 图像大小保持不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p, bias=False)
        # 每次卷积后都要经过一次标准化与激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SCNet(nn.Module):
    def __init__(self, nc=7):
        super(SCNet, self).__init__()

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
            nn.Linear(16384, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 512),
            torch.nn.ReLU()
            # nn.Linear(512, nc),
        )

        self.fc2 = torch.nn.Sequential(
            nn.Linear(2, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 512),
            torch.nn.ReLU()
        )

        self.fc3 = torch.nn.Sequential(
            nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 256),
            torch.nn.ReLU(),
            nn.Linear(256, nc)
        )

        self.fc4 = torch.nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2, x3, x4):
        gri_data = x1
        urz_data = x2
        wb_data = torch.cat([x3.float(), x4.float()], dim=1)

        gri_data = self.stage1_gri(gri_data)
        urz_data = self.stage1_urz(urz_data)

        x = torch.cat([gri_data, urz_data], axis=1)
        x = self.stage2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print("Shape after x:", x.shape)  # 打印 fc1 之后的形状

        wb_data = self.fc2(wb_data)
        # print("Shape after wb:", wb_data.shape)  # 打印 fc1 之后的形状
        x_all = torch.cat([x, wb_data], axis=1)  # 现在可以连接它们
        # print("Shape after all:",x_all.shape)  # 打印 fc1 之后的形状

        return self.fc3(x_all),self.fc4(x_all)

