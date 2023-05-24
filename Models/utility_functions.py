from typing import Tuple, List, Type, Any
from numpy import load, empty, ndarray, concatenate, array, reshape
import pickle
from sklearn.model_selection import train_test_split
import os


def load_data(use_eye_dataset: bool = False, num_features: int = 310, num_session: int = 16) -> \
        Tuple[ndarray, ndarray, Any, Any]:

    # Create output list

    label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral',
                  4: 'Happy'}  # hard-coded obtained from dataset documentation

    CURRENT_DIRECTORY = os.getcwd()

    concatenated_data = empty((0, 310))
    concatenated_labels = empty((0))

    path = os.path.join(CURRENT_DIRECTORY, "Dataset", "EEG_DE_features")

    if use_eye_dataset:
        concatenated_data = empty((0, 33))
        concatenated_labels = empty((0))
        path = os.path.join(CURRENT_DIRECTORY, "Dataset", "Eye_movement_features")

    pathFunction = lambda participant_lambda: os.path.join(path, f"{participant_lambda + 1}_123.npz")

    for session in range(num_session):
        data_npz = load(pathFunction(session))

        data = pickle.loads(data_npz['data'])
        label = pickle.loads(data_npz['label'])

        for key in data:
            for epoch in data[key]:
                if len(epoch) != num_features:
                    continue
                concatenated_data = concatenate((concatenated_data, array([epoch])))
                concatenated_labels = concatenate((concatenated_labels, array([label[key][0]])))

    # Perform data split
    x_train, x_test, y_train, y_test = train_test_split(concatenated_data,
                                                        concatenated_labels,
                                                        test_size=0.2,
                                                        random_state=42)

    # reshape the data for feeding to neural nets
    if not use_eye_dataset:
        x_train = reshape(x_train, (x_train.shape[0], 31, 10))
        x_test = reshape(x_test, (x_test.shape[0], 31, 10))

    else:
        x_train = reshape(x_train, (x_train.shape[0], 1, 33))
        x_test = reshape(x_test, (x_test.shape[0], 1, 33))

    return x_train, x_test, y_train, y_test


def transfer_weights(transfer_to, transfer_from):
    for layer in transfer_to.layers:
        if layer.name in [l.name for l in transfer_from.layers]:
            layer.set_weights(transfer_from.get_layer(layer.name).get_weights())
