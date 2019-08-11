import numpy as np
from .Morlet import Morlet

class CWavelet(object):
    def __init__(self, fs, wavelet='morlet', w0=6.0, wavelet_window=None,
        freq=None, freq_scale='log', n_freq=None, interval_freq=None):

        self.fs = fs # sampling frequency
        self.dt = 1 / self.fs # sampling interval
        self.df = 1 / 12
        self.wavelet_window = wavelet_window
        if wavelet == 'morlet':
            self.wavelet = Morlet(w0=w0) # Wavelet class
        else:
            raise ValueError('Unexpected wavelet: ' + str(wavelet))

        self.W = None
        self.W_inverse = None

        if freq is None:
            freq = (13.75, self.fs // 2)
        else:
            freq = (max(0, freq[0]), min(self.fs // 2, freq[1]))

        self.freq_period = self.get_freq_period(
            freq, freq_scale=freq_scale, n_freq=n_freq, interval=interval_freq)
        self.scale = self.wavelet.freq2scale(self.freq_period)

    def _get_freq_log(self, freq):
        # get frequencies spaced at epual intervals on log scale
        # frequencies on log scale is calculated from music keys
        k0 = int(12 * np.log2(freq[0] / 440)) # minimum key
        kn = int(12 * np.log2(freq[1] / 440)) # maximum key
        freq_period = np.zeros([kn - k0 + 1], dtype=np.float64) # initialize
        for i, k in enumerate(range(k0, kn + 1)):
            freq_period[i] = 2 ** (k * self.df) * 440 # frequency at each key
        return freq_period

    def _get_freq_linear(self, freq, n_freq=None, interval=None):
        # get frequencies spaced at epual intervals on linear scale
        if interval is not None and n_freq is not None:
            # Error
            pass
        elif interval is None and n_freq is None:
            # interval is 1
            n_freq = freq[1] - freq[0] + 1
        elif interval is not None:
            # calc nfreq from interval
            n_freq = np.ceil((freq[1] - freq[0] + 1) // interval) # ceil

        freq_period = np.linspace(freq[0], freq[1], 2 * n_freq + 1, dtype=np.float64)[1::2]
        return freq_period

    def get_freq_period(self, freq, freq_scale='log', n_freq=None, interval=None):
        if freq_scale == 'log':
            freq_period = self._get_freq_log(freq)
        elif freq_scale == 'linear':
            freq_period = self._get_freq_linear(freq, n_freq=n_freq, interval=interval)
        return freq_period

    def _get_time(self, n):
        if self.wavelet_window is None:
            wavelet_window = (-n + 1, n - 1)
        else:
            wavelet_window = self.wavelet_window
        return np.arange(wavelet_window[0], wavelet_window[1] + 1)

    def _fft_convolve(self, x1, x2):
        if x1.shape[-1] < x2.shape[-1]:
            x1, x2 = x2, x1
        x1_len = x1.shape[-1]
        x2_len = x2.shape[-1]
        s1 = np.fft.fft(x1, x1_len)
        s2 = np.fft.fft(x2, x1_len)
        y = np.fft.ifft(s1 * s2, x1_len)
        return y[..., -x2_len:]

    def _transform_via_fft(self, x, stride):
        y = self._fft_convolve(x, self.W[..., ::-1].conj())
        return y[..., ::stride]

    def transform(self, x, stride=1):

        if self.wavelet_window is None:
            n = self._get_time(x.shape[-1])
            if self.W is None or self.W.shape[-1] != 2 * n - 1:
                self.W = self.wavelet.get_W(n, self.scale, self.dt)

        # wavelet transform
        y = self._transform_via_fft(x, stride)
        return y

    def transform_inverse(self, y):
        # y: [s, n]
        # W: [s, n]

        if self.wavelet_window is None:
            n = self._get_time(y.shape[-1])
            if self.W_inverse is None or self.W_inverse.shape[-1] != 2 * n - 1:
                self.W_inverse = self.wavelet.get_W_inverse(n, self.scale, self.dt)
        xs = self._fft_convolve(y, self.W_inverse)
        x = self.df * np.sum(xs.real, axis=-2)
        return x
