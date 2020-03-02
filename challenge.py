import argparse
import hashlib

import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUMBERS_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
CODE = "f55ad55fd955e4e760211d4344737f6de1b87722012ec4bea6559fccc418ff04"
N_MFCC = 128
NUM_ROWS = 44
NUM_CHANNELS = 1


def prepare_data(file_path):
    metadata = pd.read_csv(file_path)
    features = []
    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():
        file_path = str(row["file"])
        class_label = row["digit"]
        data = extract_features(file_path)
        if data is not None:
            features.append([data, class_label])

    # Convert into a Panda dataframe
    train_data = pd.DataFrame(features, columns=['feature', 'class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    X = train_data.feature.tolist()
    y = np.array(train_data.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    return X, y


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfccsscaled = []
        for mfcc in mfccs:
            mfccsscaled.append(StandardScaler().fit_transform(mfcc.reshape(-1, 1)).reshape(mfcc.shape))
        mfccsscaled = np.array(mfccsscaled)

    except Exception as e:
        print("Error encountered while parsing file: ", file_path, e)
        return None

    if mfccsscaled.shape == (N_MFCC, NUM_ROWS):
        return mfccsscaled
    else:
        return None


def reshape_data(x):
    x = np.array(x)
    x = x.reshape(*x.shape, NUM_CHANNELS)
    return x


def main():
    parser = argparse.ArgumentParser(description='Hackathon 2020 challenge.')
    parser.add_argument('code', type=int, help='The secret code')
    parser.add_argument('path_to_model', help='The path to the ML model (tensorflow loadable)')
    args = parser.parse_args()
    if hashlib.sha256(str(args.code).encode()).hexdigest() == CODE:
        print("You've got the code!")
    else:
        print("This is not the code!")
        return

    # Predict data
    model = tf.keras.models.load_model(args.path_to_model)
    x, y = prepare_data(file_path="code.csv")
    x = reshape_data(x)

    le = LabelEncoder()
    labels = list(le.fit_transform(NUMBERS_WORDS))

    predictions = model.predict(x)
    predicted_code = [labels.index(np.argmax(i)) + 1 for i in predictions]

    predicted_code = "".join(str(i) for i in predicted_code)

    if hashlib.sha256(predicted_code.encode()).hexdigest() == CODE:
        print("You won!")
    else:
        print("You predicted", predicted_code, "try again!")
        return


if __name__ == '__main__':
    main()
