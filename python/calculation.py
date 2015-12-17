from scipy.linalg import circulant
import numpy as np

def calculation(data, m, n, activation, w):
    nrow, ncol = sorted((m, n))
    r = np.random.rand(nrow, 1)
    circul_matrix = circulant(r)
    circul_matrix = np.hstack((circul_matrix, circul_matrix[:, :ncol-nrow]))

    fft_r = np.fft.fft(r)
    fft_x = np.fft.fft(data)
    Rx = np.fft.ifft(fft_r * fft_x)
    if activation == 1:
        hx = sigmoid(Rx)
    elif activation == 2:
        hx = np.tanh(Rx)

    FP = hx
    rev_x = np.flipud(data)
    s_rev_x = np.roll(rev_x, 1, axis=0)
    if activation == 1:
        dhx = hx * (1 - hx)
    elif activation == 2:
        dhx = 1 - hx**2

    fft_s_rev_x = np.fft.fft(s_rev_x.transpose())
    fft_wT_rox = np.fft.fft((w * dhx).transpose())
    BP = np.fft.ifft((fft_s_rev_x * fft_wT_rox).transpose())

    return circul_matrix, FP, BP

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
