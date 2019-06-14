import numpy as np

class CWavelet(object):
    def __init__(self, fs):
        self.fs = fs
        self.dt = 1 / self.fs

    def _morlet0(n, w0=6.0):
        w = exp(1j * w0 * n - 0.5 * n ** 2) / np.pi ** 0.25 # [n, s]
        return w

    def _morlet(n, s, w0=6.0, is_normalize=True):
        a = self.dt / s.reshape([1, -1]) # [1, s]
        w = self._morlet0(a * n.reshape([-1, 1]), w0) # [n, s]
        if is_normalize:
            w *= a ** 0.5 # [n, s]
        return w

    def _get_freq_log(freq, dtype=np.float64):
        f0 = 12 * np.log2(s[0] / 440)
        fn = 12 * np.log2(s[1] / 440)
        f0 = int(f0)
        fn = int(fn)
        freq_period = np.zeros([fn - f0], dtype=dtype)
        for i, f in enumerate(range(f0, fn)):
            freq_period[i] = 2 ** (1 / 12) * f
        return freq_period

    def _get_freq_linear(freq, dtype=np.float64):
        if len(freq) == 2:
            nf = int(freq[1] - freq[0])
        else:
            nf = freq[2]
        freq_period = np.linspace(freq[0], freq[1], 2 * nf + 1, dtype=dtype)[1::2]
        return freq_period

    def get_freq_period(freq, freq_scale='log', dtype=np.float64):
        if freq_scale == 'log':
            freq_period = self._freq_log(freq, dtype=dtype)
        elif freq_scale == 'linear':
            freq_period = self._get_freq_linear(freq, dtype=dtype)
        return freq_period

    def freq2scale(freq_period, w0=6.0):
        l = (w0 + (2 + w0 ** 2) ** 0.5) / (4 * np.pi)
        s = l / freq_period
        return s

    def transform(x, freq=None, freq_scale='log', w0=6.0, is_normalize=True, dtype=np.complex128):
        # calc time
        m = x.shape[-1]
        n_min = -((m - 1) // 2)
        n = np.arange(n_min, n_min + m)

        # limit freq
        if freq is None:
            freq = [13.75, 22050.0]
        freq[0] = max(0, freq[0])
        freq[1] = min(self.fs // 2, freq[1])

        # calc scale
        freq_period = self.get_freq_period(freq, freq_scale=freq_scale)
        s = freq2scale(freq_period, w0=w0)

        # wavelet transform
        y = np.zeros([x.shape[:-1], s.shape[-1]], dtype=dtype)
        W = self._morlet(n, s, w0=w0, is_normalize=is_normalize)
        y = np.matmul(x, W)
        return y
