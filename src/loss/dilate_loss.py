import torch
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, alpha = 0.8, gamma = 0.01):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(outputs.device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(outputs.device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss#, loss_shape, loss_temporal


def ashift_loss(outputs, targets):
    T = outputs.size(-1)
    mean, o_mean = targets.mean(dim = -1, keepdim = True), outputs.mean(dim = -1, keepdim = True)
    # reduce the effect of mean value in softmax
    normed_tgt = targets - mean
    normed_out = outputs - o_mean
    #Note: because we need a signed distance function, we use simple negation instead of L1 distance
    loss = T * torch.mean(torch.abs(1 / T - torch.softmax((normed_tgt - normed_out), dim = -1)))

    return loss

def phase_loss(outputs, targets):
    T = outputs.size(-1)
    out_fourier = torch.fft.fft(outputs, dim = -1)
    tgt_fourier = torch.fft.fft(targets, dim = -1)

    # calculate dominant frequencies
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    # filter out the non-dominant frequencies
    mask = (tgt_fourier_sq > (T)).float()
    # guarantee the number of dominant frequencies is equal or greater than T**0.5
    topk_indices = tgt_fourier_sq.topk(k = int(T ** 0.5), dim = -1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    # guarantee that the loss function always considers the mean value
    mask[...,0] = 1.
    mask = torch.where(mask > 0, 1., 0.)
    mask = mask.bool()
    inv_mask = (~mask).float()
    inv_mask /= torch.mean(inv_mask)
    zero_error = torch.abs(out_fourier) * inv_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    loss = (torch.mean(zero_error) / 2 + torch.mean(ae)) / (T ** .5)

    return loss

def amp_loss(outputs, targets):
    T = outputs.size(-1)
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim = -1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim = -1)

    out_norm = torch.norm(outputs, dim = -1, keepdim = True)
    tgt_norm = torch.norm(targets, dim = -1, keepdim = True)
    tgt_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim = -1).real
    n_tgt_corr = tgt_corr / (tgt_norm * tgt_norm)

    ccorr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim = -1).real
    n_ccorr = ccorr / (tgt_norm * out_norm)
    loss = torch.mean(torch.abs(n_tgt_corr - n_ccorr))

    return loss


def tildeq_loss(outputs, targets, alpha = .5, gamma = .0):
    #outputs = outputs.squeeze(dim = 1)
    outputs = outputs.permute(0,2,1)
    targets = targets.permute(0,2,1)

    assert not torch.isnan(outputs).any(), "Nan value detected!"
    assert not torch.isinf(outputs).any(), "Inf value detected!"

    loss = alpha * ashift_loss(outputs, targets) \
            + (1 - alpha) * phase_loss(outputs, targets) \
            + gamma * amp_loss(outputs, targets)
    assert loss == loss, "Loss Nan!"
    return loss

