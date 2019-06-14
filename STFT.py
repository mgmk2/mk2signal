import numpy as np

class STFT(object):
    def __init__(self, wlen, window='hann'):
        self.wlen = wlen
        self.fmax = self.wlen // 2
        self.strides = 0.5
        self.noverlap = int(self.wlen * self.strides)
        if window == 'hann':
            self.wf = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.wlen) / self.wlen)
        else:
            self.wf = self.ones([self.wlen])

    def transform(self, x, dtype=None):
        # reshape x to 2D
        shape_x = x.shape
        if shape_x > 1:
            # dimension of x is 2 or more
            x = x.reshape([-1, shape_x[-1]])
            n = x.shape[0]
            shape_y = shape_x[:-1]
        else:
            # dimension of x is 1
            x = x.reshape([1, -1])
            n = 1
            shape_y = []

        # set dtype
        if dtype is None:
            dtype = x.dtype

        frame = (shape_x[-1] - self.wlen) // self.noverlap + 1
        xx = np.zeros([n, frame, self.wlen], dtype=dtype)
        for tt in range(frame):
            xx[:, tt, :] = x[:, tt * self.noverlap:tt * self.noverlap + self.wlen]
        xx = xx.reshape(shape_y + [frame, self.wlen])
        y = np.fft.rfft(xx * self.wf)
        return y

    def transform_inverse(self, y, dtype=None):
        # reshape x to 2D
        shape_y = y.shape
        if shape_y > 2:
            # dimension of x is 2 or more
            y = y.reshape([-1, *shape_y[-2:]])
            n = y.shape[0]
            shape_x = shape_y[:-2]
        else:
            # dimension of x is 1
            y = y[np.newaxis, :, :]
            n = 1
            shape_x = []

        # set dtype
        if dtype is None:
            dtype = x.dtype

        frame = y.shape[-2]
        x = np.zeros([n, (frame + 1) * self.noverlap], dtype=dtype)
        for tt in range(frame):
            x[:, tt * self.noverlap:tt * self.noverlap + self.wlen] += np.fft.irfft(y[:, tt, :])
        x = x.reshape(shape_x + [(frame + 1) * self.noverlap])
        return x
