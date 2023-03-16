import torch
from torch.autograd import Function
from utils import knn_query_cuda


def knn_query(ctx, nsample, xyz, offset, new_xyz=None, new_offset=None):
    """
    input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    output: idx: (m, nsample) -1 is placeholder, dist2: (m, nsample)
    """
    if new_xyz is None or new_offset is None:
        new_xyz = xyz
        new_offset = offset
    assert xyz.is_contiguous() and new_xyz.is_contiguous()
    m = new_xyz.shape[0]
    idx = torch.cuda.IntTensor(m, nsample).zero_()
    dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
    idx, dist2 = knn_query_cuda(m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2)
    return idx, torch.sqrt(dist2)


def grouping(idx,
             feat,
             xyz,
             new_xyz=None,
             with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    xyz = torch.cat([xyz, torch.zeros([1, 3]).to(xyz.device)], dim=0)
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, num_sample, c)

    if with_xyz:
        assert new_xyz.is_contiguous()
        mask = torch.sign(idx + 1)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)  # (m, num_sample, 3)
        grouped_xyz = torch.einsum("n s c, n s -> n s c", grouped_xyz, mask)  # (m, num_sample, 3)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat





def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: coords: (m, 3), new_xyz: (n, 3), color: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)  # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8)  # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm  # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat



def knn_query_and_group(feat,
                        xyz,
                        offset=None,
                        new_xyz=None,
                        new_offset=None,
                        idx=None,
                        nsample=None,
                        with_xyz=False
                        ):
    if idx is None:
        assert nsample is not None
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx



def farthest_point_sampling(ctx, xyz, offset, new_offset):
        """
        input: coords: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        farthest_point_sampling_cuda(b, n_max, xyz, offset.int(), new_offset.int(), tmp, idx)
        del tmp
        return idx