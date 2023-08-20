import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size
    
    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            tmp = self.downsample[0].weight.shape[1]
            self.torch_filter = torch.zeros(tmp, 1, self.kernel_size, self.kernel_size, self.kernel_size).cuda()
            self.torch_filter[:, :, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2] = 1
            residual = F.conv3d(residual, self.torch_filter, stride=self.stride, padding=0, groups = self.torch_filter.shape[0])
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out
    

class model(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=64, num_levels=4, **kwargs):
        super(model, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2 ** k for k in range(num_levels)]  # number_of_features_per_level

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create layers for channel-wise dynamics
        layers = []
        layers.append()
        self.dynamics = nn.ModuleList(layers)
        

        # create layers for cross-channel dependency

        # create layers for coarse to fine fusion 

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # Apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network
        # outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x
