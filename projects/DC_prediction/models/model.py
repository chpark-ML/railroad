import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

import projects.DC_prediction.utils.constants as C


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.generate_positional_encoding()

    def generate_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.positional_encoding[:, :x.size(-1), :].permute(0, 2, 1).unsqueeze(2).to(x.device)
        return x
    

class FusionBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, padding=0):
        super(FusionBlock, self).__init__()
        self.conv = ResidualBlock(inplanes=inplanes, planes=planes, kernel_size=kernel_size, stride=stride, padding=padding, flag_res=True)

    def forward(self, x_low, x_high, **kwargs):
        x_high = F.interpolate(x_high, scale_factor=(1, 2), mode='bilinear', align_corners=False)
        
        diff_t = x_low.size()[3] - x_high.size()[3]
        x_high = F.pad(x_high, [diff_t // 2, diff_t - diff_t // 2])

        x_concat = torch.cat([x_low, x_high], dim=1)
        x_concat = self.conv(x_concat)
        return x_concat


class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=5, stride=1, dilation=1, padding=0, flag_res=True):
        super(ResidualBlock, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.flag_res = flag_res
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                               padding_mode='zeros', bias=False)
        self.norm1 = nn.BatchNorm2d(planes, affine=True)

        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=1, stride=1, padding=0,
                               padding_mode='zeros', bias=False)
        self.norm2 = nn.BatchNorm2d(planes, affine=True)

        if self.flag_res:
            if stride != 1 or inplanes != planes * self.expansion :
                self.downsample = nn.Sequential(
                        nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * self.expansion, affine=True),
                    )
            else :
                self.downsample = None

        self.act_f = nn.GELU()

    def forward(self, x, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_f(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.flag_res:
            ## downsample the residual
            if self.downsample is not None:
                residual = self.downsample(residual)

            ## crop and residual connection
            out += residual[:, :, :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out
    

class RailModel(nn.Module):
    def __init__(self, in_planes, f_maps=32, num_levels=4, kernel=5, dilation=2, 
                 rail_type="curved", window_length=2500, out_channels=4, **kwargs):
        super(RailModel, self).__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2 ** k for k in range(num_levels)]  # number_of_features_per_level

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.inplanes = in_planes
        self.rail_type = rail_type
        self.num_channels = C.NUM_CHANNEL_MAPPER[rail_type]
        self.window_length = window_length
        self.out_channels = out_channels
        
        # TODO: This is tot done yet
        self.yaw_embedding = nn.Embedding(num_embeddings=len(C.YAW_TYPES),
                                          embedding_dim=in_planes)

        self.pos_enc = PositionalEncoding(d_model=in_planes * 2, max_seq_len=self.window_length)

        # embedding 
        self.rail_embedding = nn.Sequential(
            nn.Conv2d(1, in_planes, kernel_size=(1, kernel), stride=(1, 1), padding=(0, kernel//2), padding_mode='zeros', dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(in_planes, affine=True),
            nn.GELU(),
        )

        # create layers encoders
        dynamic_layers = []
        cross_channel_layers = []
        for idx, f in enumerate(f_maps):
            dynamic_layers.append(ResidualBlock(inplanes=2 * self.inplanes if idx == 0 else self.inplanes, 
                                                planes=f, kernel_size=(1, kernel),
                                                stride=(1, 1) if idx == 0 else (1, kernel//2), 
                                                dilation=(1, dilation), padding=(0, kernel//2 * dilation), flag_res=True))
            cross_channel_layers.append(nn.Sequential(
                ResidualBlock(inplanes=f, planes=f * self.out_channels, kernel_size=(self.num_channels, 1), stride=1, dilation=1, padding=0, flag_res=True),
                ResidualBlock(inplanes=f * self.out_channels, planes=f * self.out_channels, kernel_size=(1, 1), stride=1, dilation=1, padding=0, flag_res=True),
            ))
            self.inplanes = f

        self.time_encoder = nn.ModuleList(dynamic_layers)
        self.channel_encoder = nn.ModuleList(cross_channel_layers)

        # create decoders
        fusion_layers = []
        r_f_maps = [f for f in f_maps[::-1]]
        for f_l, f_h in zip(r_f_maps[1:], r_f_maps[:-1]):
            fusion_layers.append(FusionBlock(f_l + f_h, f_l))
        self.decoders = nn.ModuleList(fusion_layers)

        drop_p = 0.1
        self.final_conv = nn.Sequential(
            nn.Conv2d(f_maps[0], 50, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout2d(p=drop_p),
            nn.Sigmoid(),
            nn.Conv2d(50, 30, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout2d(p=drop_p),
            nn.Sigmoid(),
            nn.Conv2d(30, 15, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Dropout2d(p=drop_p),
            nn.Sigmoid(),
            nn.Conv2d(15, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor, yaw: torch.Tensor = None):
        # embedding
        x = self.rail_embedding(x)  # B, 16, C, 2500

        if yaw is not None:
            # FIXME: this code will NOT operate due to tensor dimension issue
            yaw = self.yaw_embedding(yaw)
            yaw = yaw.unsqueeze(2).unsqueeze(3).repeat((1, 1, x.size()[-2], x.size()[-1]))
            x = torch.concat(tensors=[x, yaw], dim=1)  # B, 16, C, 2500

        # positional encoding
        x = self.pos_enc(x)  # B, 16, C, 2500

        # encoder part
        c_encoders_features = []
        for t_encoder, c_encoder in zip(self.time_encoder, self.channel_encoder):
            x = t_encoder(x)  # B, 32, C, 1250
            x_c = c_encoder(x)  # B, 32, 1, 1250
            x_c = x_c.view(x_c.shape[0], x_c.shape[1] // self.out_channels, self.out_channels, x_c.shape[3])  # B, 32 / 4, 4, 1250
            c_encoders_features.insert(0, x_c)
        c_encoders_features = c_encoders_features[1:]

        # decoder part
        for decoder, encoder_feature in zip(self.decoders, c_encoders_features):
            x_c = decoder(encoder_feature, x_c)

        x_c = self.final_conv(x_c)
        return x_c
