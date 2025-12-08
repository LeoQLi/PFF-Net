import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .encode import EncodeNet
from .base import PConv, PBlock, Conv1D, Attention, cos_angle


class MLPNet(nn.Module):
    def __init__(self,
                 d_aug=64,
                 d_code=0,
                 d_mid=128,
                 d_out=64,
                 n_mid=1,
                 with_grad=False,
            ):
        super(MLPNet, self).__init__()
        assert n_mid > 3
        dims = [d_aug + d_code] + [d_mid for _ in range(n_mid)] + [d_out]
        self.skip_in = [n_mid // 2 + 1]    # TODO

        self.with_grad = with_grad
        if with_grad:
            dims += [3]

        self.mlp_1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, d_aug, 1),
        )

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                assert dims[l + 1] > d_aug + d_code
                out_dim = dims[l + 1] - d_aug - d_code  # TODO
            else:
                out_dim = dims[l + 1]
            lin = nn.Conv1d(dims[l], out_dim, 1)

            setattr(self, "lin" + str(l), lin)

    def forward(self, pos, code=None):
        """
            pos: (B, C, N)
            code: (B, C, N)
        """
        pos = self.mlp_1(pos)
        x = torch.cat([pos, code], dim=1)

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, pos, code], dim=1)

            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                xx = x if self.with_grad else None
                x = F.relu(x)

        return x, xx


class PointEncoder(nn.Module):
    def __init__(self, knn, use_w=True):
        super(PointEncoder, self).__init__()
        self.use_w = use_w

        self.encodeNet = EncodeNet(
                                knn=knn,
                                num_convs=2,
                                in_channels=3,
                                conv_channels=24,
                            )
        d_code = self.encodeNet.out_channels   # 60

        d_out = 64
        self.mlp_net = MLPNet(d_code=d_code, d_mid=128, d_out=d_out, n_mid=6)

        self.conv_1 = Conv1D(d_code + d_out, 128)

        dim_p = 256
        self.pblock_1 = PBlock(128, dim_p)
        self.pblock_m = PBlock(dim_p, dim_p)   # Note: we add another block compared to the paper.
        self.pblock_2 = PBlock(dim_p, dim_p)

        self.pconv_o = PConv(dim_p, dim_p)
        self.conv_2 = Conv1D(dim_p, dim_p)
        self.conv_3 = Conv1D(dim_p, 128)

        if self.use_w:
            self.alpha = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            self.beta = nn.Parameter(torch.Tensor(1,), requires_grad=True)
            torch.nn.init.ones_(self.alpha.data)
            torch.nn.init.ones_(self.beta.data)

    def forward(self, pos, knn_idx):
        """
            pos:  (B, 3, N)
            knn_idx: (B, N, K)
        """
        num_pcl = pos.shape[-1]

        y = self.encodeNet(x=pos, pos=pos, knn_idx=knn_idx)       # (B, C, N)
        y0, _ = self.mlp_net(pos=pos, code=y)                     # (B, C, N)
        y = torch.cat([y, y0], dim=1)                             # (B, C, N)
        y = self.conv_1(y)

        if self.use_w:
            ### compute distance weights from points to its center point (ref: FKAConv, POCO)
            dist = torch.sqrt((pos.detach() ** 2).sum(dim=1))              # (B, N)
            dist_w = torch.sigmoid(-self.alpha * dist + self.beta)
            dist_w_s = dist_w.sum(dim=1, keepdim=True)                     # (B, 1)
            dist_w_s = dist_w_s + (dist_w_s == 0) + 1e-6
            dist_w = (dist_w / dist_w_s * dist.shape[1]).unsqueeze(1)      # (B, 1, N)
        else:
            dist_w = torch.ones_like(pos[:, 0:1, :])

        s = [1, 1, 2, 3]
        ### decrease the number of points and add the features
        yp = self.pblock_1(y, num_pcl//2**s[0], dist_w=dist_w)    # (B, C, N/2)
        yp = self.pblock_m(yp, num_pcl//2**s[1], dist_w=dist_w)
        yp = self.pblock_2(yp, num_pcl//2**s[2], dist_w=dist_w)

        num_out = yp.shape[-1]
        yp = self.pconv_o(yp, num_out, dist_w=dist_w)             # (B, C, N')
        yp = self.conv_3(self.conv_2(yp))                         # (B, C, N')
        return yp, y[:, :, :num_out], dist_w[:, :, :num_out]


class Network(nn.Module):
    def __init__(self, encode_knn=16):
        super(Network, self).__init__()
        self.encode_knn = encode_knn

        self.pointEncoder = PointEncoder(knn=self.encode_knn, use_w=True)

        pos_dim = 64
        out_dim = 128 + pos_dim
        self.attention = Attention(q_dim=128, kv_dim=128, dim=pos_dim)

        dim_p = 128
        self.pblock_1 = PBlock(out_dim, dim_p)
        self.pblock_2 = PBlock(dim_p, dim_p)

        self.pconv_o = PConv(dim_p, dim_p)
        self.conv_1 = Conv1D(dim_p, 128)

        self.conv_n = Conv1D(128, 128)
        self.conv_w = nn.Conv1d(128, 1, 1)
        self.mlp_n  = nn.Linear(128, 3)
        self.mlp_nn = nn.Linear(128, 3)

    def forward(self, pcl_pat, mode_test=False):
        """
            pcl_pat: (B, N, 3)
        """
        normal, weights, neighbor_normal = None, None, None

        _, knn_idx, _ = knn_points(pcl_pat, pcl_pat, K=self.encode_knn+1, return_nn=False)  # (B, N, K+1)

        ### Encoder
        pcl_pat = pcl_pat.transpose(2, 1)                                 # (B, 3, N)
        x, xx, wd = self.pointEncoder(pcl_pat, knn_idx=knn_idx[:,:,1:self.encode_knn+1])    # (B, C, n)

        xc = self.attention(q=x, kv=xx, wd=wd)

        num_out = x.shape[-1]
        yp = self.pblock_1(xc, num_out, dist_w=wd)
        yp = self.pblock_2(yp, num_out, dist_w=wd)

        yp = self.pconv_o(yp, num_out, dist_w=wd)
        feat = self.conv_1(yp)

        ### Output
        weights = 0.01 + torch.sigmoid(self.conv_w(feat))                 # (B, 1, n)
        feat_w = self.conv_n(feat * weights * wd[..., :num_out])          # (B, C, n)
        normal = self.mlp_n(feat_w.max(dim=2, keepdim=False)[0])          # (B, 3)
        neighbor_normal = self.mlp_nn(feat.transpose(2, 1))               # (B, n, 3)

        if mode_test:
            return normal, neighbor_normal

        return normal, weights, neighbor_normal


    def get_loss(self, q_target, q_pred, ne_target=None, ne_pred=None, pred_weights=None, pcl_in=None,
                normal_loss_type=['ms_euclidean', 'sin']):
        """
            q_target: query point normal, (B, 3)
            q_pred: query point normal, (B, 3)
            ne_target: patch point normal, (B, N, 3)
            ne_pred: patch point normal, (B, N, 3)
            pred_weights: patch point weight, (B, 1, N)
            pcl_in: input (noisy) point clouds, (B, N, 3)
        """
        _device, _dtype = q_target.device, q_target.dtype
        weight_loss = torch.zeros(1, device=_device, dtype=_dtype)
        normal_loss1 = torch.zeros(1, device=_device, dtype=_dtype)
        normal_loss2 = torch.zeros(1, device=_device, dtype=_dtype)
        consistency_loss1 = torch.zeros(1, device=_device, dtype=_dtype)
        consistency_loss2 = torch.zeros(1, device=_device, dtype=_dtype)
        bias_loss = torch.zeros(1, device=_device, dtype=_dtype)
        angle_loss = torch.zeros(1, device=_device, dtype=_dtype)

        ### center point normal
        if q_pred is not None:
            if 'mse_loss' in normal_loss_type:
                normal_loss += 0.25 * F.mse_loss(q_pred, q_target)
            if 'ms_euclidean' in normal_loss_type:
                normal_loss1 = 0.1 * torch.min((q_pred - q_target).pow(2).sum(1),
                                                (q_pred + q_target).pow(2).sum(1)).mean()
            if 'ms_oneminuscos' in normal_loss_type:
                cos_ang = cos_angle(q_pred, q_target)
                normal_loss += 1.0 * (1 - torch.abs(cos_ang)).pow(2).mean()
            if 'sin' in normal_loss_type:
                normal_loss2 = 0.1 * torch.linalg.norm(torch.cross(q_pred, q_target, dim=-1), ord=2, dim=1).mean()

        ### neighboring point normal
        if ne_pred is not None:
            num_out = ne_pred.shape[1]
            weights = pred_weights.squeeze() if pred_weights is not None else torch.ones_like(ne_pred)[:,:,0]
            ne_target = ne_target[:, :num_out, :].contiguous()
            batch_size, n_points, _ = ne_target.shape

            if 'mse_loss' in normal_loss_type:
                consistency_loss += 0.5 * torch.mean(weights * (ne_pred - ne_target).pow(2).sum(2))
            if 'ms_euclidean' in normal_loss_type:
                consistency_loss1 = 0.4 * torch.mean(weights * torch.min((ne_pred - ne_target).pow(2).sum(2),
                                                                        (ne_pred + ne_target).pow(2).sum(2)))
            if 'ms_oneminuscos' in normal_loss_type:
                cos_ang = cos_angle(ne_pred.view(-1, 3), ne_target.view(-1, 3)).view(batch_size, n_points)
                consistency_loss += torch.mean(weights * (1 - torch.abs(cos_ang)).pow(2))
            if 'sin' in normal_loss_type:
                consistency_loss2 = 0.4 * torch.mean(weights *
                                    torch.linalg.norm(torch.cross(ne_pred.view(-1, 3), ne_target.view(-1, 3), dim=-1).view(batch_size, n_points, 3),
                                                ord=2, dim=2))

        ### compute true_weight by fitting distance
        if pred_weights is not None:
            pred_weights = pred_weights.squeeze()
            num_out = pred_weights.shape[1]
            pcl_local = pcl_in[:, :num_out, :] - pcl_in[:, 0:1, :]                                      # (B, N, 3)
            scale = (pcl_local ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0]      # (B, 1, 1)
            pcl_local = pcl_local / scale

            ### the distances between the neighbor points and ground truth tangent plane
            gamma = 0.3
            thres_d = 0.05 ** 2
            normal_dis = torch.bmm(q_target.unsqueeze(1), pcl_local.transpose(2, 1)).pow(2).squeeze()   # (B, N), dis^2
            sigma = torch.mean(normal_dis, dim=1) * gamma + 1e-5                                        # (B,)
            threshold_matrix = torch.ones_like(sigma) * thres_d                                         # (B,)
            sigma = torch.where(sigma < thres_d, threshold_matrix, sigma)                               # (B,), all sigma >= thres_d
            true_weights_plane = torch.exp(-1 * torch.div(normal_dis, sigma.unsqueeze(-1)))             # (B, N), -dis/mean(dis') -> (-∞, 0) -> (0, 1)

            true_weights = true_weights_plane
            weight_loss = 1.0 * (true_weights - pred_weights).pow(2).mean()

        losses = (normal_loss1, normal_loss2, consistency_loss1, consistency_loss2, weight_loss, bias_loss, angle_loss)
        return sum(losses), losses


