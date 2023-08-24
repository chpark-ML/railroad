import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

import projects.DC_prediction.utils.constants as C
import projects.DC_prediction.models.modules as M
    

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

        # embedding layer
        self.embedding = M.EmbeddingBlock(d_model=in_planes, kernel=7, max_seq_len=self.window_length)

        # create encoders
        dynamic_layers = []
        cross_channel_layers = []
        for idx, f in enumerate(f_maps):
            dynamic_layers.append(M.ResidualBlock(inplanes=self.inplanes, 
                                                  planes=f, kernel_size=(1, kernel),
                                                  stride=(1, 1) if idx == 0 else (1, kernel//2), 
                                                  dilation=(1, dilation), padding=(0, kernel//2 * dilation), flag_res=True))
            cross_channel_layers.append(nn.Sequential(
                M.ResidualBlock(inplanes=f, planes=f * self.out_channels, kernel_size=(self.num_channels, 1), stride=1, dilation=1, padding=0, flag_res=True),
            ))
            self.inplanes = f
        self.time_encoder = nn.ModuleList(dynamic_layers)
        self.channel_encoder = nn.ModuleList(cross_channel_layers)

        # catch dependency between [y, t'] (e.g., [4, 125])
        self.global_encoder = M.MultiHeadSelfAttention(n_featuremap=f_maps[-1], n_heads=4,  d_k=f_maps[-1] * 2)

        # create decoders
        fusion_layers = []
        r_f_maps = [f for f in f_maps[::-1]]
        for f_l, f_h in zip(r_f_maps[1:], r_f_maps[:-1]):
            fusion_layers.append(M.FusionBlock(f_l + f_h, f_l))
        self.decoders = nn.ModuleList(fusion_layers)

        # classifier
        drop_p = 0.01
        self.classifier = M.Classifier(f_maps[0], drop_p=drop_p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor, yaw: torch.Tensor = None):
        # embedding
        x = self.embedding(x, yaw)  # torch.Size([4, 16, 37, 2000])
        
        # encoder part
        """
        torch.Size([4, 32, 4, 2500])
        torch.Size([4, 64, 4, 1250])
        torch.Size([4, 128, 4, 625])
        torch.Size([4, 256, 4, 313])
        """
        c_encoders_features = []
        for t_encoder, c_encoder in zip(self.time_encoder, self.channel_encoder):
            x = t_encoder(x)  
            x_c = c_encoder(x)  
            x_c = x_c.view(x_c.shape[0], x_c.shape[1] // self.out_channels, self.out_channels, x_c.shape[3])  
            c_encoders_features.insert(0, x_c)

        # prepare inputs for decoder
        x_c = self.global_encoder(x_c)  # B, 256, 4, 125
        c_encoders_features = c_encoders_features[1:]

        # decoder part
        """
        torch.Size([4, 128, 4, 625])
        torch.Size([4, 64, 4, 1250])
        torch.Size([4, 32, 4, 2500])
        """
        for decoder, encoder_feature in zip(self.decoders, c_encoders_features):
            x_c = decoder(encoder_feature, x_c)

        # classifier
        x_c = self.classifier(x_c)

        return x_c
