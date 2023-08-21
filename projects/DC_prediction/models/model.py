import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, padding = 0):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=False)
        self.norm = nn.InstanceNorm2d(planes, affine=False)
        self.act_f = nn.ReLU(inplace=True)

    def forward(self, x_low, x_high, **kwargs):
        x_high = F.interpolate(x_high, size=(x_low.size()[-2:]))
        x_concat = torch.cat([x_low, x_high], dim=1)
        x_concat = self.conv(x_concat)
        x_concat = self.norm(x_concat)
        x_concat = self.act_f(x_concat)

        return x_concat
    

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, dilation=1, padding = 0, flag_res = True):
        super(ResidualBlock, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.flag_res = flag_res
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=False)
        self.norm1 = nn.InstanceNorm2d(planes, affine=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, padding_mode='replicate', bias=False)
        self.norm2 = nn.InstanceNorm2d(planes, affine=False)

        self.act_f = nn.ReLU(inplace=True)

        if self.flag_res == True:
            if stride != 1 or inplanes != planes * self.expansion :
                self.downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.InstanceNorm2d(planes * self.expansion, affine=False),
                    )
            else :
                self.downsample = None

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_f(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.flag_res == True:
            ## downsample the residual
            if self.downsample is not None:
                residual = self.downsample(residual)

            ## crop and residual connection
            out += residual[:, :, :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out
    

class model(nn.Module):
    def __init__(self, in_planes, f_maps=32, num_levels=4, num_channels=37, out_channels=4, **kwargs):
        super(model, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2 ** k for k in range(num_levels)]  # number_of_features_per_level

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.inplanes = in_planes
        self.num_channels = num_channels
        self.out_channels = out_channels

        # embedding 
        self.embedding = nn.Sequential(
            nn.Conv2d(1, in_planes, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), padding_mode='replicate', dilation=1, groups=1, bias=False),
            nn.InstanceNorm3d(in_planes, affine=False),
            nn.ReLU(),
        )

        # create layers encoders
        dynamic_layers = []
        cross_channel_layers = []
        for idx, f in enumerate(f_maps):
            if idx == 0:
                dynamic_layers.append(ResidualBlock(inplanes=self.inplanes, planes=f, kernel_size=(1, 5), stride=(1, 1), 
                                                    dilation=1, padding=(0, 2), flag_res=True))
            else:
                dynamic_layers.append(ResidualBlock(inplanes=self.inplanes, planes=f, kernel_size=(1, 5), stride=(1, 2), 
                                                    dilation=1, padding=(0, 2), flag_res=True))
            cross_channel_layers.append(
                ResidualBlock(inplanes=f, planes=f, kernel_size=(num_channels, 1), stride=1, dilation=1, 
                              padding=0, flag_res=True))
            self.inplanes = f
        
        self.time_encoder = nn.ModuleList(dynamic_layers)
        self.channel_encoder = nn.ModuleList(cross_channel_layers)

        # create decoders
        fusion_layers = []
        r_f_maps = [f // self.out_channels for f in f_maps[::-1]]
        for f_l, f_h in zip(r_f_maps[1:], r_f_maps[:-1]):
            fusion_layers.append(FusionBlock(f_l + f_h, f_l))
        self.decoders = nn.ModuleList(fusion_layers)
        self.final_conv = nn.Conv2d(f_maps[0] // self.out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # embedding
        x = self.embedding(x)  # B, 16, 37, 2500

        # encoder part
        c_encoders_features = []
        for t_encoder, c_encoder in zip(self.time_encoder, self.channel_encoder):
            x = t_encoder(x)  # B, 32, 37, 1250
            x_c = c_encoder(x)  # B, 32, 1, 1250
            x_c = x_c.view(x_c.shape[0], x_c.shape[1] // self.out_channels, self.out_channels, x_c.shape[3])
            c_encoders_features.insert(0, x_c)
        c_encoders_features = c_encoders_features[1:]

        # decoder part
        for decoder, encoder_feature in zip(self.decoders, c_encoders_features):
            x_c = decoder(encoder_feature, x_c)

        x_c = self.final_conv(x_c)
        
        return x_c