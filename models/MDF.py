import torch
from torch import nn
import torch.nn.functional as F

class DMC(nn.Module):
    def __init__(self, channels, factor=16, dilations=[1, 3, 5]):
        super(DMC, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=d, dilation=d, stride=1)
            for d in dilations
        ])

        self.dilated_concat_conv = nn.Conv2d((channels // self.groups) * len(dilations), channels // self.groups, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        dilated_features = []
        for conv in self.dilated_convs:
            dilated_x = conv(group_x)
            dilated_features.append(dilated_x)

        dilated_concat = torch.cat(dilated_features, dim=1)

        dilated_concat = self.dilated_concat_conv(dilated_concat)  # 使用 1x1 卷积调整维度
        dilated_concat = dilated_concat.view(b * self.groups, -1, h, w)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = dilated_concat
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, -1, h * w)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, -1, h * w)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class AEFusion(nn.Module):
    def __init__(self, in_channels_x, in_channels_y, out_channels):
        super(AEFusion, self).__init__()
        # 1x1 convolutions to adjust channel dimensions
        self.adjust_x = nn.Conv2d(in_channels_x, out_channels, kernel_size=1)
        self.adjust_y = nn.Conv2d(in_channels_y, out_channels, kernel_size=1)
        # 1x1 convolution to adjust the final result channels
        self.adjust_final = nn.Conv2d( 3 * out_channels, out_channels, kernel_size=1)

    def forward(self, x, y):
        # Adjust channel dimensions of x and y
        x = self.adjust_x(x)
        y = self.adjust_y(y)

        # Ensure spatial dimensions of x and y match
        if x.shape[2:]!= y.shape[2:]:
            raise ValueError(f"Spatial dimensions of x {x.shape[2:]} and y {y.shape[2:]} must match.")

        # Concatenate x and y along the channel dimension
        result = torch.cat((x, y), dim=1)

        # Apply softmax along the channel dimension
        result_softmax = torch.softmax(result, dim=1)

        # Multiply the softmax result with y
        y1 = torch.mul(result_softmax[:, :y.size(1), :, :], y)

        # Concatenate x and y1 along the channel dimension
        x1 = torch.cat((x, y1), dim=1)

        # Apply average pooling along spatial dimensions
        avg_pool = F.adaptive_avg_pool2d(x1, (1, 1))

        # Apply max pooling along spatial dimensions
        max_pool = F.adaptive_max_pool2d(x1, (1, 1))

        # Add pooled results and apply sigmoid activation
        pooled_result = avg_pool + max_pool
        activated_result = torch.sigmoid(pooled_result)

        # Upscale activated_result to match spatial dimensions of y1
        activated_result_upsampled = F.interpolate(
            activated_result, size=y1.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate activated_result and y1 along the channel dimension
        final_result = torch.cat((activated_result_upsampled, y1), dim=1)

        # Adjust the final result channels to out_channels
        final_result = self.adjust_final(final_result)
        return final_result



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([x_avg, x_max], dim=1)
        spatial_attn = self.conv(x_concat)
        return spatial_attn


class EMA(nn.Module):
    def __init__(self, channels, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = x2.view(B, -1, H, W)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim_x, dim_y, reduction=8):
        super(CGAFusion, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim_x + dim_y, reduction)
        self.pa = PixelAttention(dim_x + dim_y)
        self.conv = nn.Conv2d(dim_x + dim_y, dim_x + dim_y, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.adjust_x = nn.Conv2d(dim_x, dim_x + dim_y, 1, bias=False)
        self.adjust_y = nn.Conv2d(dim_y, dim_x + dim_y, 1, bias=False)

    def forward(self, x, y):
        x = self.adjust_x(x)
        y = self.adjust_y(y)

        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


# ==========================Core Module================================
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# ==================================================================
class MDF(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(MDF, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.EMA1 = EMA(channels=64)
        self.dmc1 = DMC(channels=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.EMA2 = EMA(channels=128)
        self.dmc2 = DMC(channels=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.EMA3 = EMA(channels=256)
        self.dmc3 = DMC(channels=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.fusion2 = CGAFusion(dim_x=64, dim_y=64, reduction=8)
        self.fusion1 = AEFusion(in_channels_x=128, in_channels_y=128, out_channels=256)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x11 = self.dmc1(x1)

        x2 = self.Maxpool(x11)
        x2 = self.Conv2(x2)
        x2 = self.dmc2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dmc3(x3)

        d3 = self.Up3(x3)
        d3 = self.EMA2(d3)
        d3 = self.fusion1(d3, x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.EMA1(d2)
        d2 = self.fusion2(d2, x1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1, dim=1)

        return d1



