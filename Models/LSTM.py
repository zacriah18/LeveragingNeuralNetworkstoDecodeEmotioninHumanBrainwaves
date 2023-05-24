from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Flatten, Input, concatenate


def create_single_modal_LSTM() -> Model:
    input1 = Input(shape=(31, 10), name="input_1")
    conv1 = Conv1D(filters=32, kernel_size=(1,), activation='relu', name="Conv_1")(input1)
    dropout1 = Dropout(0.2)(conv1)
    lstm1 = LSTM(64)(dropout1)
    dense1 = Dense(64, activation='relu')(lstm1)
    dropout3 = Dropout(0.2)(dense1)
    flatten = Flatten()(dropout3)
    output = Dense(5, activation='softmax')(flatten)

    return Model(inputs=input1, outputs=output)


def create_dual_modal_LSTM() -> Model:
    # First Modality
    input1 = Input(shape=(31, 10), name="input_1")
    conv1 = Conv1D(filters=32, kernel_size=(1,), activation='relu', name="Conv_1")(input1)
    dropout1 = Dropout(0.2)(conv1)
    lstm1 = LSTM(64)(dropout1)

    # Second Modality
    input2 = Input(shape=(1, 33), name="input_2")  # change the shape as per your second input
    conv2 = Conv1D(filters=32, kernel_size=(1,), activation='relu', name="Conv_2")(input2)
    dropout2 = Dropout(0.2)(conv2)
    lstm2 = LSTM(64)(dropout2)

    # Concatenate both branches
    merge = concatenate([lstm1, lstm2])

    dense1 = Dense(64, activation='relu')(merge)
    dropout3 = Dropout(0.2)(dense1)
    flatten = Flatten()(dropout3)
    output = Dense(5, activation='softmax')(flatten)

    return Model(inputs=[input1, input2], outputs=output)
