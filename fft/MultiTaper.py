import numpy as np
import numpy.linalg as LA

class MultiTaper(object):
    def __init__(self, fs, resolution):
        self.fs = fs
        self.resolution = resolution
        self.N = None
        self.W = None

    def get_taper(self, N):
        r = 2 * self.resolution * self.fs / N
        k = np.arange(N, dtype=np.float64)
        kl = k[:, np.newaxis] - k[np.newaxis, :]
        kl[np.eye(N, dtype=bool)] = 1
        Phi = np.sin(np.pi * r / self.fs * kl) / kl
        Phi[np.eye(N, dtype=bool)] = np.pi * r / self.fs
        _, W = LA.eig(Phi)
        L = int(2 * r) - 1
        if L == 1:
            return W[:, :L][:, np.newaxis].real
        return W[:, :L].real

    def transform(self, x, L=None):
        N = x.shape[-1]
        if self.N is None or self.N != N:
            self.N = N
            self.W = self.get_taper(N)
        if L is None:
            L = self.W.shape[-1]
        xk = x[..., np.newaxis, :] * self.W[:, :L].T
        Xk = np.fft.rfft(xk)
        X = np.mean(np.abs(Xk) ** 2, axis=-2)
        return X
