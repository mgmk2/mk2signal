import numpy as np

class Mel(object):
    def __init__(self, window, fs):
        self.window = window
        self.fs = fs
        self.fft_freq = fs / window * np.arange(0, window / 2 + 1)
        self.filterbank = None

    def freq2mel(self, f):
        return 2595 * np.log10(1 + f / 700)

    def mel2freq(self, m):
        return 700 * (np.power(10, m / 2595) - 1)

    def build_filterbank(self, num_filter):
        mel_low = 0
        mel_high = self.freq2mel(self.fs / 2)
        mel = np.linspace(mel_low, mel_high, num_filter + 2)
        freq = self.mel2freq(mel)

        filterbank = np.zeros([num_filter, self.window // 2 + 1], dtype=np.float32)

        for i in range(1, num_filter + 1):
            for j in range(self.window // 2 + 1):
                if freq[i - 1] < self.fft_freq[j] < freq[i]:
                    filterbank[i - 1, j] = (self.fft_freq[j] - freq[i - 1]) / (freq[i] - freq[i - 1])
                elif self.fft_freq[j] == freq[i]:
                    filterbank[i - 1, j] = 1.0
                elif freq[i] < self.fft_freq[j] < freq[i + 1]:
                    filterbank[i - 1, j] = -(self.fft_freq[j] - freq[i + 1]) / (freq[i + 1] - freq[i])

        self.filterbank = filterbank
        return filterbank

    def filt(self, x, num_filter):
        if self.filterbank is None or self.filterbank.shape[0] != num_filter:
            filterbank = self.build_filterbank(num_filter)
        else:
            filterbank = self.filterbank
        y = np.matmul(x, filterbank.T)
        return y
