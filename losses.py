import torch
import random
import torch.nn.functional as F
import math


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def compute_diversity(pred_traj):
    """
    计算预测轨迹间的多样性（基于平均成对距离）
    """
    if pred_traj.size(1) < 2:
        return 0.0  # 单样本无法计算多样性
    diff = pred_traj.unsqueeze(1) - pred_traj.unsqueeze(0)
    distances = torch.norm(diff, p=2, dim=-1).mean()
    return distances


def displacement_error1(pred_traj, pred_traj_gt, loss_mask=None, mode='mean'):
    if loss_mask is None:
        loss_mask = torch.ones_like(pred_traj_gt)
    loss = torch.norm(pred_traj - pred_traj_gt, p=2, dim=-1) * loss_mask
    if mode == 'sum':
        return torch.sum(loss)
    else:
        return torch.mean(loss)

def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def mfp_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """
    MFP (K=1) loss using 2D Gaussian negative log-likelihood.

    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 5). Predicted params:
                 [mu_x, mu_y, sigma_x, sigma_y, rho]
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth trajectory.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be 'sum', 'average', 'raw'

    Output:
    - loss: NLL loss depending on mode
    """
    seq_len, batch = pred_traj.size(0), pred_traj.size(1)

    # Extract predicted Gaussian parameters
    mu = pred_traj[..., 0:2]                      # (seq_len, batch, 2)
    sigma = F.softplus(pred_traj[..., 2:4])       # ensure positive std
    rho = torch.tanh(pred_traj[..., 4])           # ensure -1 < rho < 1

    # Ground truth
    x = pred_traj_gt[..., 0]
    y = pred_traj_gt[..., 1]
    mu_x = mu[..., 0]
    mu_y = mu[..., 1]
    sx = sigma[..., 0]
    sy = sigma[..., 1]

    eps = 1e-6
    one_minus_rho2 = 1 - rho**2 + eps

    norm_x = (x - mu_x) / sx
    norm_y = (y - mu_y) / sy
    z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    log_exp = z / (2 * one_minus_rho2)

    log_det = torch.log(2 * math.pi * sx * sy * torch.sqrt(one_minus_rho2))
    nll = log_exp + log_det   # shape: (seq_len, batch)

    # Apply mask
    nll = nll.permute(1, 0) * loss_mask  # shape: (batch, seq_len)

    if mode == 'sum':
        return nll.sum()
    elif mode == 'average':
        return nll.sum() / loss_mask.sum()
    elif mode == 'raw':
        return nll.sum(dim=1)  # per sample in batch


def mfp_loss_k1(pred_trajs, pred_traj_gt, loss_mask, mode='average'):
    """
    Multiple Futures Prediction (MFP) loss based on minimum ADE.

    Inputs:
    - pred_trajs: Tensor of shape (K, seq_len, batch, 2). K predicted futures.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth future.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: 'sum', 'average', or 'raw'

    Output:
    - loss: Minimum ADE loss over K predictions per sample.
    """
    seq_len, batch, _ = pred_trajs.size()

    # Permute for easier broadcasting: (K, batch, seq_len, 2)
    pred = pred_trajs.permute(2, 1, 3)  # (K, batch, seq_len, 2)
    gt = pred_traj_gt.permute(1, 0, 2).unsqueeze(0)  # (1, batch, seq_len, 2)
    mask = loss_mask.unsqueeze(0).permute(0, 1, 2)  # (1, batch, seq_len)

    # Squared error over time
    error = ((pred - gt) ** 2).sum(dim=3)  # (K, batch, seq_len)
    masked_error = error * mask  # apply mask

    # Average Displacement Error (ADE)
    ade = masked_error.sum(dim=2) / mask.sum(dim=2)  # (K, batch)

    # Take the minimum ADE for each sample
    min_ade, _ = ade.min(dim=0)  # (batch,)

    if mode == 'sum':
        return min_ade.sum()
    elif mode == 'average':
        return min_ade.mean()
    elif mode == 'raw':
        return min_ade


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
