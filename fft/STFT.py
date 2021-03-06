import numpy as np

class STFT(object):
    def __init__(self, wlen, window='hann'):
        self.wlen = wlen
        self.fmax = self.wlen // 2
        self.strides = 0.5
        self.noverlap = int(self.wlen * self.strides)
        self.window = window
        self.wf = self.get_window(window)

    def get_window(self, window, dtype=None):
        if dtype is None:
            dtype = np.float64

        if window == 'hann':
            wf = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.wlen) / self.wlen)
        else:
            print('Unknown window name: ' + window)
            wf = self.ones([self.wlen])
        return wf.astype(dtype)

    def transform(self, x, dtype=None):
        # set dtype
        if dtype is None:
            dtype = x.dtype

        # get window func
        wf = self.get_window(dtype=dtype)

        frame = (x.shape[-1] - self.wlen) // self.noverlap + 1
        xx = np.zeros([*x.shape[:-1], frame, self.wlen], dtype=dtype)
        for tt in range(frame):
            xx[..., tt, :] = x[..., tt * self.noverlap:tt * self.noverlap + self.wlen]
        y = np.fft.rfft(xx * wf)
        return y

    def transform_inverse(self, y, dtype=None):
        # set dtype
        if dtype is None:
            dtype = y.dtype

        frame = y.shape[-2]
        x = np.zeros([*y.shape[:-2], (frame + 1) * self.noverlap], dtype=dtype)
        xx = np.fft.irfft(y)
        for tt in range(frame):
            x[..., tt * self.noverlap:tt * self.noverlap + self.wlen] += xx[..., tt, :]
        return x
