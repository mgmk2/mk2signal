# mk2signal

## Overview
mk2signal is a simple signal processing module for Python3. mk2signals contains continuous wavelet transform (CWT), short time fourier transform (STFT), etc.

## Description
Modules containing signal processing tools, for example CWT and STFT, are already exists. However, they have some problems (scaling frequencies and/or power) and difficulty to understand returned values. So I developped mk2signal for practical use and also for my studying.

## Requirement
* numpy

## contained tools
* Short Time Fourier Transform and Inverse Transform (Hann window only)
* Mel Filterbank
* Continuous Wavelet Transform and Inverse Transform (Morlet Wavelet only)
* Multi-Taper Method (Power Spectrum Density estimation)
