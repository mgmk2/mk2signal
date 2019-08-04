import numpy as np
from scipy.signal import fftconvolve

class CWavelet(object):
    def __init__(self, fs):
        self.fs = fs # sampling frequency
        self.dt = 1 / self.fs # sampling interval
        self.W = None # Wavelet matrix

    def _morlet0(self, n, w0=6.0):
        # calc morlet func
        w = np.exp(1j * w0 * n - 0.5 * n ** 2) / np.pi ** 0.25 # [n, s]
        return w

    def morlet(self, n, s, w0=6.0, is_normalize=True):
        # calc morlet and normalize
        a = self.dt / s.reshape([-1, 1]) # [s, 1]
        w = self._morlet0(a * n.reshape([1, -1]), w0) # [s, n]
        if is_normalize:
            w *= a # [s, 1]
        return w

    def _get_freq_log(self, freq, dtype=np.float64):
        # get frequencies spaced at epual intervals on log scale
        # frequencies on log scale is calculated from music keys
        k0 = int(12 * np.log2(freq[0] / 440)) # minimum key
        kn = int(12 * np.log2(freq[1] / 440)) # maximum key
        freq_period = np.zeros([kn - k0 + 1], dtype=dtype) # initialize
        for i, k in enumerate(range(k0, kn + 1)):
            freq_period[i] = 2 ** (k / 12) * 440 # frequency at each key
        return freq_period

    def _get_freq_linear(self, freq, n_freq=None, interval=None, dtype=np.float64):
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

        freq_period = np.linspace(freq[0], freq[1], 2 * n_freq + 1, dtype=dtype)[1::2]
        return freq_period

    def get_freq_period(self, freq, freq_scale='log', n_freq=None, interval=None, dtype=np.float64):
        if freq_scale == 'log':
            freq_period = self._get_freq_log(freq, dtype=dtype)
        elif freq_scale == 'linear':
            freq_period = self._get_freq_linear(freq, n_freq=n_freq, interval=interval, dtype=dtype)
        return freq_period

    def _freq2scale_morlet(self, freq_period, w0=6.0):
        return (w0 + (2 + w0 ** 2) ** 0.5) / (4 * np.pi) / freq_period

    def freq2scale(self, freq_period, wavelet='morlet', w0=6.0):
        if wavelet == 'morlet':
            s = self._freq2scale_morlet(freq_period, w0=w0)
        else:
            print('Warning:: Unexpected wavelet: ' + str(wavelet) + '. Scales are inverse of frequencies.')
            s = 1 / freq_period
        return s

    def _transform_via_fft(self, x, n, s, stride, dtype):
        y = fftconvolve(x.astype(dtype)[np.newaxis, ...], self.W.conj(), mode='valid')
        return y[..., ::stride]

    def transform(self, x, stride=1, wavelet='morlet', w0=6.0, wavelet_window=None,
            freq=None, freq_scale='log', n_freq=None, interval_freq=None,
            is_normalize=True, dtype=np.complex128):

        # set freq limit
        if freq is None:
            freq = [13.75, 22050.0]
        freq[0] = max(0, freq[0])
        freq[1] = min(self.fs // 2, freq[1])

        # set scale
        freq_period = self.get_freq_period(
            freq, freq_scale=freq_scale, n_freq=n_freq, interval=interval_freq)
        s = self.freq2scale(freq_period, wavelet=wavelet, w0=w0)

        # set wavelet window
        if wavelet_window is None:
            wavelet_window = (-x.shape[-1] + 1, x.shape[-1] - 1)
        n = np.arange(wavelet_window[0], wavelet_window[1] + 1)
        length = wavelet_window[1] - wavelet_window[0] + 1

        # set wavelet
        if wavelet == 'morlet':
            self.W = self.morlet(n, s, w0=w0, is_normalize=is_normalize)
        else:
            raise ValueError('Unexpected wavelet: ' + str(wavelet))

        # wavelet transform
        y = self._transform_via_fft(x, n, s, stride, dtype)
        return y, freq_period
