import numpy as np

class Morlet(object):
    def __init__(self, w0=6.0):
        self.w0 = w0
        self.c = (1 + np.exp(-self.w0 ** 2) - 2 * np.exp(-0.75 * self.w0 ** 2)) ** 0.5
        self.n = None
        self.s = None
        self.dt = None

    def is_update_W(self, n, s, dt):
        return self.n is None or self.s is None or self.dt is None \
            or not np.array_equal(self.n, n) or not np.array_equal(self.s, s) \
            or self.dt != dt

    def _get_W(self, t):
        k = np.exp(-0.5 * self.w0 ** 2)
        W = self.c / np.pi ** 0.25 *\
            np.exp(-0.5 * t ** 2) * (np.exp(1j * self.w0 * t) - k) # [n, s]
        return W

    def get_W(self, n, s, dt):
        if self.is_update_W(n, s, dt):
            self.n = n
            self.s = s
            self.dt = dt
            scale_factor = dt / s.reshape([-1, 1]) # [s, 1]
            t = scale_factor * n.reshape([1, -1])
            self.W = scale_factor * self._get_W(t)
        return self.W

    def get_W_inverse(self, n, s, dt):
        return np.pi ** 0.25 / self.c * self.get_W(n, s, dt)

    def freq2scale(self, freq):
        return (self.w0 + (2 + self.w0 ** 2) ** 0.5) / (4 * np.pi) / freq

    def relative_power(self, n, s, dt):
        scale_factor = dt / s # [s, 1]
        w = self.get_W(n, s, dt)
        return np.sum(np.abs(w) ** 2, axis=-1) / scale_factor
