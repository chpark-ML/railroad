import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

import projects.DC_prediction.utils.constants as C


class Classifier(nn.Module):
    def __init__(self, inplanes, drop_p=0.2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(inplanes, 1, kernel_size=1, bias=True),
            )

    def forward(self, x):
        return self.classifier(x)


class EmbeddingBlock(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(EmbeddingBlock, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.yaw_embedding = nn.Embedding(num_embeddings=len(C.YAW_TYPES), embedding_dim=d_model)
        self.positional_encoding = self.generate_positional_encoding()

    def generate_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x, yaw=None):
        pe = self.positional_encoding[:, :x.size(-1), :].permute(0, 2, 1).unsqueeze(2).to(x.device)  # [B, f, 1, t']
        if yaw is not None:
            yaw = self.yaw_embedding(yaw).unsqueeze(2).unsqueeze(3)  # [B, f, 1, 1]
            return x + pe+ yaw
        return  x + pe
    

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
        # self.norm1 = nn.LayerNorm(planes, eps=1e-6)
        self.norm1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=1, stride=1, padding=0,
                               padding_mode='zeros', bias=False)
        # self.norm2 = nn.LayerNorm(planes, eps=1e-6)
        self.norm2 = nn.BatchNorm2d(planes)

        if self.flag_res:
            if stride != 1 or inplanes != planes * self.expansion :
                self.downsample = nn.Sequential(
                        # nn.Linear(inplanes, planes * self.expansion),
                        # nn.LayerNorm(planes * self.expansion, eps=1e-6),
                        nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                                  padding_mode='zeros', bias=False),
                        nn.BatchNorm2d(planes * self.expansion),
                    )
            else :
                self.downsample = None

        self.act_f = nn.LeakyReLU(inplace=True)

    def forward(self, x, **kwargs):
        residual = x
        out = self.conv1(x)
        # out = out.permute(0, 2, 3, 1)
        out = self.norm1(out)
        # out = out.permute(0, 3, 1, 2)
        out = self.act_f(out)

        out = self.conv2(out)
        # out = out.permute(0, 2, 3, 1)
        out = self.norm2(out)
        # out = out.permute(0, 3, 1, 2)

        if self.flag_res:
            ## downsample the residual
            if self.downsample is not None:
                # residual = residual.permute(0, 2, 3, 1)
                residual = self.downsample(residual)
                # residual = residual.permute(0, 3, 1, 2)

            ## crop and residual connection
            out += residual[:, :, :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_featuremap, n_heads = 4,  d_k = 16):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv2d(n_featuremap, self.num_heads * self.d_k , kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv2d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv2d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv2d(self.num_heads * self.d_k, n_featuremap, kernel_size=1, bias=False)
        # self.norm = nn.LayerNorm(n_featuremap, eps=1e-6)
        self.norm = nn.BatchNorm2d(n_featuremap)
        self.act_f = nn.LeakyReLU(inplace=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        total_key_depth = width * height

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        logits = logits / math.sqrt(keys.size(-1))
        weights = self.softmax(logits)  # B X (N) X (N/p)

        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, -1).permute(0, 3, 1, 2)
        out = self.output_conv(out)
        # out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        # out = out.permute(0, 3, 1, 2)

        """ residual """
        out = self.gamma * out
        out = out + x
        out = self.act_f(out)

        return out

