import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


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
