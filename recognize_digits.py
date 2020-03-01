# Python3 script to recognize speech from text

import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

# Note: N_MFCC max is 128
N_MFCC = 128
NUM_ROWS = 44
NUM_CHANNELS = 1
# Model params.
NUM_EPOCHS = 300
NUM_BATCH_SIZE = 128
MODEL_SAVE_PATH = 'saved_model/weights0103.games.basic_cnn.hdf5'


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


def split_test_train_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    num_labels = y.shape[1]
    return num_labels, x_test, x_train, y_test, y_train


def build_model(num_columns, num_rows, num_channels, num_labels):
    # Construct model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_columns, num_rows, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model


def main():
    x, y = prepare_data(file_path="train.csv")

    print('Finished feature extraction from', len(y), 'files')

    num_labels, x_test, x_train, y_test, y_train = split_test_train_data(x, y)

    model = build_model(N_MFCC, NUM_ROWS, NUM_CHANNELS, num_labels)

    # Display model architecture summary
    model.summary()

    # Calculate pre-training accuracy
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100 * score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH,
                                   verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(x_train, y_train,
              batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test), callbacks=[checkpointer],
              verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    # Predict data
    x, _ = prepare_data(file_path="code.csv")
    x = reshape_data(x)

    model.predict(x)


if __name__ == '__main__':
    main()
