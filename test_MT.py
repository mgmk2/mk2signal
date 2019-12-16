import numpy as np
import matplotlib.pyplot as plt
from fft.MultiTaper import MultiTaper

fs = 512
N = 512
f = 64.5
x = np.sin(2 * np.pi * f * np.arange(N) / N)
x += np.sin(2 * np.pi * 2 * f * np.arange(N) / N)
noise = 2 * np.random.rand(N) - 1
x += 0.1 * noise
#x = np.zeros(N)
#x[N // 2] = 1
MT = MultiTaper(fs, 1)
X_MT = MT.transform(x)
H_MT = np.fft.rfft(MT.W.T)

X = np.abs(np.fft.rfft(x)) ** 2 / N

hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N)/N)
X_hann = np.abs(np.fft.rfft(hann * x)) ** 2 / N

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(hann)

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(10 * np.log10(X))
ax.plot(10 * np.log10(X_hann))
ax.plot(10 * np.log10(X_MT))

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(X)
ax.plot(X_hann)
ax.plot(X_MT)
plt.show()

W_hann = np.abs(np.fft.rfft(hann)) ** 2 / N
W_MT = np.sum(np.abs(np.fft.rfft(MT.W.T)) ** 2, axis=0) / N

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(10 * np.log10(W_hann))
ax.plot(10 * np.log10(W_MT))
plt.show()
