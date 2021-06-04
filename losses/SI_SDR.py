  
import torch
# from itertools import permutations

def ScaleInvariantSDRLoss(x, s, eps=1e-8):
    """
    Calculate training loss for time domain signals
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2_norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(f"Dimention mismatch when calculate si-snr, {x.shape} vs {s.shape}")

    # Subtract mean
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    
    # Element-wise multiply and normalize
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2_norm(s_zm, keepdim=True)**2 + eps)
    
    # Compute SDR 
    si_sdr = 20 * torch.log10(eps + l2_norm(t) / (l2_norm(x_zm - t) + eps))
    return si_sdr