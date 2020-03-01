# Python3 script to recognize speech from text

import tensorflow as tf
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Python3 code for fft of wav and fft to wav

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# Python3 code to rename multiple files in a directory or folder

import os
import csv
import uuid
import random

from recognize_digits import prepare_data, reshape_data
from recognize_fft import get_fft_of_file


# Note: N_MFCC max is 128

N_MFCC = 128
NUM_ROWS = 44
NUM_CHANNELS = 1
# Model params.
NUM_EPOCHS = 300
NUM_BATCH_SIZE = 128
MODEL_SAVE_PATH = 'saved_model/weights2902.best.basic_cnn.hdf5'
SECRET_DICT = {
        1: 2,
        2: 3,
        3: 4,
        4: 6,
        5: 7,
        6: 1,
        7: 5,
        8: 9,
        9: 8,
    }
NUMBERS_TO_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
NUMBERS_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def generate_challenge(length=5):
    sample_rate = 4400
    frequencies = range(200, 1000, 100)
    power = [5, 3.5, 2.6, 1.8, 1.5, 10, 2.1, 1.15, 1.3]
    t = np.linspace(0, length, sample_rate * length)

    y = 1 / power[0] * np.sin(100 * 2 * np.pi * t)
    for i, frequency in enumerate(frequencies):
        y = y + 1 / power[i+1] * np.sin(frequency * 2 * np.pi * t)

    file_name = f'code_{length}sec.wav'
    wavfile.write(file_name, sample_rate, y)


def pick_files_for_code(code):
    csv_columns = ['file', 'digit']
    csv_file_name = "code.csv"
    number_of_files = 1
    for digit in code:
        word = NUMBERS_TO_WORDS[digit]
        folder_name = "/Users/liorpollak/Downloads/data-speech_commands_v0.02/" + word
        try:
            with open(csv_file_name, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                file_list = os.listdir(folder_name)
                random.shuffle(file_list)
                file_list = file_list[:number_of_files]
                for file_name in file_list:
                    #new_file_name = "{}_{}.wav".format(word, str(uuid.uuid4()))
                    #os.rename(folder_name + "/" + file_name, folder_name + "/" + new_file_name)
                    writer.writerow({"file": folder_name + "/" + file_name, "digit": word})
        except IOError as e:
            print("I/O error", e)


def main():
    code = [2, 4, 4, 5, 7]
    length = 5
    generate_challenge(length)
    fft_out, frq_label = get_fft_of_file(f'code_5sec.wav')
    frq_label = frq_label / 100
    fft_out = fft_out / 1000

    plt.plot(frq_label[:5500], fft_out[:5500], 'r')
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.grid()
    plt.show()

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Predict data
    #pick_files_for_code(code)
    x, y = prepare_data(file_path="code.csv")
    x = reshape_data(x)
    #z = np.zeros((y.shape[0], 9))
    #z[:y.shape[0],:y.shape[1]] = y

    le = LabelEncoder()
    labels = list(le.fit_transform(NUMBERS_WORDS))

    predictions = model.predict(x)
    predicted_code = []
    #print(model.evaluate(x, z))
    for i in predictions:
        #print(max(i), labels.index(np.argmax(i)) + 1)
        predicted_code.append(labels.index(np.argmax(i)) + 1)

    if predicted_code == code:
        print("You won!")
    else:
        print("You predicted", predicted_code, "try again!")


if __name__ == '__main__':
    main()
