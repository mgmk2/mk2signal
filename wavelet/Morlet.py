import numpy as np

class Morlet(object):
    def __init__(self, w0=6.0):
        self.w0 = w0
        self.c = (1 + np.exp(-self.w0 ** 2) - 2 * np.exp(-0.75 * self.w0 ** 2)) ** 0.5

    def _get_W(self, t):
        k = np.exp(-0.5 * self.w0 ** 2)
        w = self.c / np.pi ** 0.25 * \
            np.exp(-0.5 * t ** 2) * (np.exp(1j * self.w0 * t) - k) # [n, s]
        return w

    def get_W(self, n, s, dt):
        norm_factor = dt / s.reshape([-1, 1]) # [s, 1]
        t = norm_factor * n.reshape([1, -1])
        return self._get_W(t) * norm_factor

    def get_W_inverse(self, n, s, dt):
        return self.get_W(n, s, dt) * np.pi ** 0.25 / self.c

    def freq2scale(self, freq):
        return (self.w0 + (2 + self.w0 ** 2) ** 0.5) / (4 * np.pi) / freq
