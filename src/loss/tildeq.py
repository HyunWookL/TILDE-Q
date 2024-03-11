import torch
#from . import soft_dtw
#from . import path_soft_dtw 

PI = 3.141592653589793

def amp_loss(outputs, targets):
    #outputs = B, T, 1 --> B, 1, T
    B,_, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim = -1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim = -1)

    out_norm = torch.norm(outputs, dim = -1, keepdim = True)
    tgt_norm = torch.norm(targets, dim = -1, keepdim = True)

    #calculate normalized auto correlation
    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim = -1).real
    auto_corr = torch.cat([auto_corr[...,-(T-1):], auto_corr[...,:T]], dim = -1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    # calculate cross correlation
    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim = -1).real
    cross_corr = torch.cat([cross_corr[...,-(T-1):], cross_corr[...,:T]], dim = -1)
    nac_out = cross_corr / (tgt_norm * out_norm)
    
    loss = torch.mean(torch.abs(nac_tgt - nac_out))
    return loss


def ashift_loss(outputs, targets):
    B, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim = -1)))


def phase_loss(outputs, targets):
    B, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim = -1)
    tgt_fourier = torch.fft.fft(targets, dim = -1)
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    mask = (tgt_fourier_sq > (T)).float()
    topk_indices = tgt_fourier_sq.topk(k = int(T**0.5), dim = -1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[...,0] = 1.
    mask = torch.where(mask > 0, 1., 0.)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    out_fourier_sq = (torch.abs(out_fourier.real) + torch.abs(out_fourier.imag))
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    phase_loss = (torch.mean(zero_error) + torch.mean(ae)) / (T ** .5)
    return phase_loss


def tildeq_loss(outputs, targets, alpha = .5, gamma = .0, beta = .5):
    outputs = outputs.permute(0,2,1)
    targets = targets.permute(0,2,1)
    assert not torch.isnan(outputs).any(), "Nan value detected!"
    assert not torch.isinf(outputs).any(), "Inf value detected!"
    B,_, T = outputs.shape
    l_ashift = ashift_loss(outputs, targets)
    l_amp = amp_loss(outputs, targets)
    l_phase = phase_loss(outputs, targets)
    loss = alpha * l_ashift + (1 - alpha) * l_phase + gamma * l_amp

    assert loss == loss, "Loss Nan!"
    return loss


if __name__ == "__main__":
    import numpy as np
    import torch
    import os
    path = './results/Exchange_96_336_nbeats_custom_ftS_sl96_ll48_pl336_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_fourier_0'
    gt = torch.Tensor(np.load(os.path.join(path, 'true.npy')))
    pred = torch.Tensor(np.load(os.path.join(path, 'pred.npy')))
    tildeq_loss(pred, gt)
