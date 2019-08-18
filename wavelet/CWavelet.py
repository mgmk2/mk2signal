import numpy as np
from .Morlet import Morlet

def fft_convolve(x1, x2):
    if x1.shape[-1] < x2.shape[-1]:
        x1, x2 = x2, x1
    x1_len = x1.shape[-1]
    x2_len = x2.shape[-1]
    s1 = np.fft.fft(x1, x1_len)
    s2 = np.fft.fft(x2, x1_len)
    y = np.fft.ifft(s1 * s2, x1_len)
    return y[..., -x2_len:]

class CWavelet(object):
    def __init__(self, fs, wavelet='morlet', w0=6.0, window_len=None,
        freq_range=None):

        self.fs = fs # sampling frequency
        self.dt = 1 / self.fs # sampling interval
        self.n_per_octave = 12
        self.window_len_init = window_len
        self.window_len = window_len
        self.W = None
        if wavelet == 'morlet':
            self.wavelet = Morlet(w0=w0) # Wavelet class
        else:
            raise ValueError('Unexpected wavelet: ' + str(wavelet))

        if freq_range is None:
            freq_range = (13.75, self.fs // 2)
        else:
            freq_range = (max(0, freq_range[0]), min(self.fs // 2, freq_range[1]))
        self.freq_period = self.get_freq_period(freq_range)
        self.scale = self.wavelet.freq2scale(self.freq_period)

    def _get_freq_log(self, freq_range):
        # get frequencies spaced at epual intervals on log scale
        # frequencies on log scale is calculated from music keys
        k0 = int(self.n_per_octave * np.log2(freq_range[0] / 440)) # minimum key
        kn = int(self.n_per_octave * np.log2(freq_range[1] / 440)) # maximum key
        k = np.arange(k0, kn + 1, dtype=np.float64)
        freq_period = 440 * 2 ** (k / self.n_per_octave) # frequency
        return freq_period

    def get_freq_period(self, freq_range):
        freq_period = self._get_freq_log(freq_range)
        return freq_period

    def _get_window_len(self, n):
        if self.window_len_init is None:
            window_len = 2 * n - 1
        else:
            window_len = self.window_len_init
        return window_len

    def _transform_via_fft(self, x, strides):
        y = fft_convolve(x[..., np.newaxis, :], self.W[..., ::-1].conj())
        return y[..., ::strides]

    def _transform(self, x, strides):
        y = self._transform_via_fft(x, strides)
        return y

    def _update_W(self, N):
        window_len = self._get_window_len(N)
        if self.W is None or self.window_len != window_len:
            self.window_len = window_len
            self.n = np.arange(-window_len // 2, (window_len - 1) // 2)
            t = self.n * self.dt
            self.W = self.wavelet.get_W(t, self.scale)

    def transform(self, x, strides=1):
        self._update_W(x.shape[-1])
        y = self.dt * self._transform(x, strides)
        return y

    def transform_inverse(self, y):
        self._update_W(y.shape[-1])
        xs = fft_convolve(y, self.W) / self.scale.reshape([-1, 1])
        coefficient = 2 * np.log(2) * self.dt / self.n_per_octave / self.wavelet.C
        x = coefficient * np.sum(xs.real, axis=-2)
        return x
