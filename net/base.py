import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def cos_angle(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, F, N)
    :param  idx:    (B, M, k)
    :return (B, F, M, k)
    """
    B, F, N = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(2).expand(B, F, M, N)
    idx = idx.unsqueeze(1).expand(B, F, M, k)

    return torch.gather(x, dim=3, index=idx)


def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    _, knn_idx, _ = knn_points(pos, query, K=k+offset, return_nn=False)
    return knn_idx[:, :, offset:]


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x


class PConv(nn.Module):
    def __init__(self, input_dim, output_dim, with_mid=True):
        super(PConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_mid = with_mid

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_mid)

        if with_mid:
            self.conv_mid = Conv1D(input_dim*2, input_dim//2, with_bn=True)
            self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)

    def forward(self, x, num_out, dist_w=None):
        """
            x: (B, C, N)
        """
        BS, _, N = x.shape

        if dist_w is None:
            y = self.conv_in(x)
        else:
            y = self.conv_in(x * dist_w[:,:,:N])        # (B, C*2, N)

        feat_g = torch.max(y, dim=2, keepdim=True)[0]   # (B, C*2, 1)
        if self.with_mid:
            feat_g = self.conv_mid(feat_g)              # (B, C/2, 1)

        x = torch.cat([x[:, :, :num_out],
                    feat_g.view(BS, -1, 1).repeat(1, 1, num_out),
                    ], dim=1)

        x = self.conv_out(x)
        return x


class PBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(PBlock, self).__init__()
        self.pconv_11 = PConv(dim1, dim2)
        self.pconv_12 = PConv(dim2, dim2)

    def forward(self, x, num_out, dist_w=None):
        num_in = x.shape[-1]
        y11 = self.pconv_11(x, num_in, dist_w=dist_w)             # (B, C, N)
        y12 = self.pconv_12(y11, num_out, dist_w=dist_w)          # (B, C, N')
        y12 = y12 + y11[:, :, :num_out]                           # (B, C, N')
        return y12


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim, dim):
        super(Attention, self).__init__()
        self.conv_q = nn.Conv1d(q_dim, dim, 1)
        self.conv_k = nn.Conv1d(kv_dim, dim, 1)
        self.conv_v = nn.Conv1d(kv_dim, dim, 1)
        self.conv_attn = nn.Conv1d(dim, dim, 1)

    def forward(self, q, kv, wd):
        query = self.conv_q(q)                        # (B, C, n)
        key   = self.conv_k(kv)                       # (B, C, n)
        value = self.conv_v(kv)
        attn  = self.conv_attn(query + key)
        # attn = torch.softmax(attn, dim=1)
        agg   = attn * value * wd
        xc = torch.cat([agg, q], dim=1)
        return xc
