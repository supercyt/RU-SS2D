import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.sub_modules.vmamba import SS2D


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def warp_feature(x, record_len, pairwise_t_matrix):
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        i = 0  # ego
        neighbor_feature = warp_affine_simple(batch_node_features[b],
                                              t_matrix[i, :, :, :],
                                              (H, W))
        out.append(neighbor_feature)

    out = torch.cat(out, dim=0)

    return out

def cosine_sim_map(a, b, eps=1e-6):
    # a, b: (C, H, W)
    a_n = F.normalize(a.view(a.size(0), -1), dim=0, eps=eps).view_as(a)
    b_n = F.normalize(b.view(b.size(0), -1), dim=0, eps=eps).view_as(b)
    # per-pixel cosine similarity
    sim = (a_n * b_n).sum(dim=0, keepdim=True)  # (1, H, W)
    return sim.clamp(-1.0, 1.0)

def tukey_biweight(r, c=0.3):
    # r >= 0, (1 - (r/c)^2)^2 when r<c else 0
    x = (r / c).clamp(min=0.0)
    w = (1 - x**2).clamp(min=0.0)**2
    return w

def robust_weights_per_pixel(ego_feat, nbr_feats, c=0.3, eps=1e-6):
    """
    ego_feat: (C, H, W)
    nbr_feats: (K, C, H, W)
    return: w: (K, 1, H, W) in [0,1]
    """
    K, C, H, W = nbr_feats.shape
    sims = []
    for j in range(K):
        sim = cosine_sim_map(ego_feat, nbr_feats[j])  # (1,H,W)
        sims.append(sim)
    sims = torch.cat(sims, dim=0)  # (K, H, W)
    r = (1 - sims).clamp(min=0.0)  # 残差
    w = tukey_biweight(r, c=c).unsqueeze(1)  # (K,1,H,W)
    return w.clamp(min=eps)

def pose_scalar_gate(t_mat_row, alpha=2.0, beta=0.4, gamma=0.8, eta=0.0):
    """
    t_mat_row: (K, 2, 3) or (K, 3, 3) normalized affine to ego；只用相对平移&旋转近似。
    这里简单地从仿射里取平移与旋转近似（yaw），给出一个 [0,1] 的标量 gate。
    """
    K = t_mat_row.shape[0]
    # 取平移
    tx = t_mat_row[:, 0, 2]
    ty = t_mat_row[:, 1, 2]
    trans = torch.sqrt(tx**2 + ty**2)  # 像素域位移幅值

    # 取旋转近似（从 2x2 子矩阵）
    a = t_mat_row[:, 0, 0]; b = t_mat_row[:, 0, 1]
    yaw = torch.atan2(b, a).abs()  # 近似相对角

    gate = torch.sigmoid(alpha - beta*trans - gamma*yaw)  # (K,)
    return gate  # (K,)

def valid_overlap_gate_ones(nbr_num, H, W, t_row, device):
    """
    可选的更准的 gate：warp 全 1 掩膜得到有效覆盖率（标量）。
    计算量会略增，默认不启用。
    """
    ones = torch.ones((nbr_num, 1, H, W), device=device)
    warped = warp_affine_simple(ones, t_row, (H, W))  # (K,1,H,W)
    # 因为出界会变成 0，取均值即可近似覆盖率
    cover = warped.mean(dim=(2,3)).squeeze(1)  # (K,)
    # 把覆盖率挤压进 (0,1)，也可以直接用 cover
    gate = cover.clamp(0,1)
    return gate

class RobustSS2DFusion(nn.Module):
    """
    一阶段鲁棒 SS2D：
    - 像素级 M-估计权重 (w_j(u))
    - 邻车标量先验 gate (g_j)
    - 送入 SS2D 前做 alpha_j(u) 归一缩放
    计算量：与原法同阶，仅增加若干逐元素操作
    """
    def __init__(self, feature_dims, c_tukey=0.3, use_mask_prior=False,
                 alpha=2.0, beta=0.4, gamma=0.8, eta=0.0):
        super().__init__()
        self.ssd = SS2D(d_model=feature_dims, channel_first=True, forward_type="v05_noz")
        self.c_tukey = c_tukey
        self.use_mask_prior = use_mask_prior
        self.alpha = alpha; self.beta = beta; self.gamma = gamma; self.eta = eta

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        x: (B_total, C, H, W) 
        record_len: (B,)
        normalized_affine_matrix: (B, L, L, 2, 3)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(x, record_len)   # List[Tensor]: len=B, 每个是 (N_i, C, H, W)
        out = []

        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]  # (N,N,2,3)
            # 统一 warp 到 ego (i=0)
            i = 0
            warped = warp_affine_simple(split_x[b], t_matrix[i, :, :, :], (H, W))  # (N,C,H,W)
            K = warped.shape[0]

            # 计算像素级鲁棒权重（相对 ego）
            ego = warped[0]                # (C,H,W)
            nbr = warped                   # (K,C,H,W)
            w_pix = robust_weights_per_pixel(ego, nbr, c=self.c_tukey)  # (K,1,H,W)

            # 标量先验 gate：由相对仿射 t_matrix[i, j]
            g = pose_scalar_gate(t_matrix[i], alpha=self.alpha,
                                 beta=self.beta, gamma=self.gamma, eta=self.eta).to(warped.device).float()  # (K,)
            if self.use_mask_prior:
                g_mask = valid_overlap_gate_ones(K, H, W, t_matrix[i], warped.device)  # (K,)
                g = (g * g_mask).clamp(1e-6, 1.0)

            # 组合 α_j(u) = softmax_j( log(w) + log(g) )
            # 展开为 (K,1,H,W)
            g_map = g.view(K, 1, 1, 1).expand_as(w_pix)
            logit = (w_pix + 1e-8).log() + (g_map + 1e-8).log()
            # softmax over K (邻车维)
            alpha = torch.softmax(logit.view(K, -1), dim=0).view_as(logit)  # (K,1,H,W)

            # 把 α 应到 token 上（像素级、邻车级），再走 SS2D
            # 形状变换： (K,C,H,W) -> (C,K,H*W) -> (1,C,K,H*W)
            weighted = (warped * alpha)  # (K,C,H,W)
            tokens = weighted.view(K, C, -1).permute(1, 0, 2).unsqueeze(0).contiguous()  # (1,C,K,H*W)

            tokens_out = self.ssd(tokens)  # (1,C,K,H*W)
            h = tokens_out.squeeze(0).permute(1, 0, 2).view(K, C, W, H).contiguous()[0, ...]  # 取 ego
            out.append(h)

        return torch.stack(out)