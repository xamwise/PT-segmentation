from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layernorm1d import LayerNorm1d
import einops

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
    
    

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3),
                                      LayerNorm1d(3),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(LayerNorm1d(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, out_planes // share_planes),
                                      LayerNorm1d(out_planes // share_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(x_k, p, o, new_xyz=p, new_offset=o,
                                                nsample=self.nsample, with_xyz=True)
        x_v, _ = pointops.knn_query_and_group(x_v, p, o, new_xyz=p, new_offset=o,
                                              idx=idx, nsample=self.nsample, with_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = x_k - x_q.unsqueeze(1) + einops.reduce(p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes)
        w = self.linear_w(r_qk)     # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum("n t s i, n t i -> n s i",
                         einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes), w)
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x