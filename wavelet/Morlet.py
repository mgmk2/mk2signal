import numpy as np

class Morlet(object):
    def __init__(self, w0=6.0):
        self.w0 = w0
        self.Cw = (1 + np.exp(-self.w0 ** 2) - 2 * np.exp(-0.75 * self.w0 ** 2)) ** (-0.5)
        self.C = self._get_C()

    def _get_C(self):
        # calc admissible constant
        wmax = 1000
        dw = 0.01
        w = np.arange(-wmax, wmax + 1, dw)
        Wf = self._get_W_fourier(w)
        C = 2 * np.pi * dw * np.sum(np.abs(Wf) ** 2 / np.abs(w))
        return C

    def _get_W_fourier(self, w):
        # calc fourier transform of wavelet
        k = np.exp(-0.5 * self.w0 ** 2)
        Wf = self.Cw / np.pi ** 0.25 *\
            (np.exp(-0.5 * (w - self.w0) ** 2) - k * np.exp(-0.5 * w ** 2))
        return Wf

    def _get_W(self, t):
        # calc wavelet
        k = np.exp(-0.5 * self.w0 ** 2)
        W = self.Cw / np.pi ** 0.25 *\
            np.exp(-0.5 * t ** 2) * (np.exp(1j * self.w0 * t) - k) # [n, s]
        return W

    def get_W(self, n, s):
        # calc wavelet at scale s
        t = n.reshape([1, -1]) / s.reshape([-1, 1])
        W = self._get_W(t) / s.reshape([-1, 1]) ** 0.5
        return W

    def freq2scale(self, freq):
        # calc scales from frequencies.
        # center freq wf is given by the solution of below:
        #   wf = w0 / (1 - exp(-wf * w0))
        # if w0 > 5, wf is nearly equal to w0.

        wf = self.w0
        for i in range(100):
            w = wf
            wf = self.w0 / (1 - np.exp(-w * self.w0))
            if np.abs(wf - w) < 1.0e-8:
                break
        return w / (2 * np.pi * freq)
