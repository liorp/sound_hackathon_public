# Python3 code for fft of wav and fft to wav

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np


def generate_sine_wave(sample_rate=4400, frequency=440, length=5):
    t = np.linspace(0, length, sample_rate * length)  # Produces a `length` second Audio-File
    y = np.sin(frequency * 2 * np.pi * t)  # Has frequency of `frequency`
    return y


def get_fft_of_file(file_name):
    rate, data = wavfile.read(file_name)
    fft_out = fft(data)
    d = int(len(fft_out) / 2)  # you only need half of the fft list (real signal symmetry)
    k = np.arange(d)
    T = len(k) / rate
    frq_label = k / (2 * T)
    frq_values = np.abs(fft_out[:d])
    return frq_values, frq_label


def main():
    length = 5
    sample_rate = 500
    frequency = 200
    y = generate_sine_wave(sample_rate, frequency, length)
    file_name = f'sine_{frequency}hz_{length}sec.wav'
    wavfile.write(file_name, sample_rate, y)
    fft_out, frq_label = get_fft_of_file(file_name)

    plt.plot(frq_label, fft_out, 'r')
    plt.show()

    rate, data = wavfile.read(file_name)
    power_spectrum, freqencies_found, time, image_axis = plt.specgram(data[:], Fs=sample_rate)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    main()
