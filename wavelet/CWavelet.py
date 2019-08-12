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

    def _get_window(self, n):
        return np.arange(-n // 2, (n - 1) // 2)

    def _get_range(self, n):
        if self.wavelet_window is None:
            m = 2 * n - 1
        else:
            m = self.wavelet_window
        return self._get_window(m)

    def _fft_convolve(self, x1, x2):
        if x1.shape[-1] < x2.shape[-1]:
            x1, x2 = x2, x1
        x1_len = x1.shape[-1]
        x2_len = x2.shape[-1]
        s1 = np.fft.fft(x1, x1_len)
        s2 = np.fft.fft(x2, x1_len)
        y = np.fft.ifft(s1 * s2, x1_len)
        return y[..., -x2_len:]

    def _transform_via_fft(self, x, strides):
        y = self._fft_convolve(x[..., np.newaxis, :], self.W[..., ::-1].conj())
        return y[..., ::strides]

    def _transform_direct(self, x, strides):
        nw = self.W.shape[-1]
        p = (x.ndim - 1) * [(0, 0)] + [(nw // 2, nw // 2)]
        x2 = np.pad(x, p, mode='constant')
        n2 = x2.shape[-1]
        num_strides = (n2 - nw) // strides + 1
        y = np.zeros([*list(x.shape[:-1]), self.W.shape[0], num_strides], np.complex128)
        for i in range(num_strides):
            xi = x2[..., np.newaxis, i * strides:i * strides + nw]
            y[..., :, i] = np.sum(xi.astype(np.complex128) * self.W, axis=-1)
        return y

    def _transform(self, x, strides):
        Nmax = max(x.shape[-1], self.W.shape[-1])
        Nmin = min(x.shape[-1], self.W.shape[-1])
        cost_fft = Nmax * np.log(Nmax)
        cost_direct = Nmin * max(1, x.shape[-1] // strides)
        if cost_fft < cost_direct:
            print('use fft convolution')
            y = self._transform_via_fft(x, strides)
        else:
            print('use direct convolution')
            y = self._transform_direct(x, strides)
        return y

    def transform(self, x, strides=1):
        self.n = self._get_range(x.shape[-1])
        if self.W is None or self.W.shape[-1] != self.n.shape[-1]:
            self.W = self.wavelet.get_W(self.n, self.scale, self.dt)
        # wavelet transform
        y = self._transform(x, strides)
        return y

    def transform_inverse(self, y):
        # y: [s, n]
        # W: [s, n]

        self.n = self._get_range(y.shape[-1])
        if self.W_inverse is None or self.W_inverse.shape[-1] != self.n[-1]:
            self.W_inverse = self.wavelet.get_W_inverse(self.n, self.scale, self.dt)
        xs = self._fft_convolve(y, self.W_inverse)
        x = self.df * np.sum(xs.real, axis=-2)
        return x

    def relative_power(self, N):
        n = self._get_window(N)
        return self.wavelet.relative_power(n, self.scale, self.dt)
