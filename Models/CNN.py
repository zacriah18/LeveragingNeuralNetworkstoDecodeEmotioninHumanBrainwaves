from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.models import Model


def create_single_modal_CNN() -> Model:
    input1 = Input(shape=(31, 10), name="input_brain")
    conv1 = Conv1D(filters=32, kernel_size=1, activation='relu', name="Conv_brain")(input1)
    pool1 = MaxPooling1D(pool_size=1, name="pool_brain")(conv1)
    flat1 = Flatten(name="flatten_brain")(pool1)
    dense1 = Dense(units=64, activation='relu')(flat1)
    output = Dense(units=5, activation='softmax')(dense1)

    return Model(inputs=input1, outputs=output)


def create_dual_modal_CNN() -> Model:
    # First modality
    input1 = Input(shape=(31, 10), name="input_brain")
    conv1 = Conv1D(filters=32, kernel_size=1, activation='relu', name="Conv_brain")(input1)
    pool1 = MaxPooling1D(pool_size=1, name="pool_brain")(conv1)
    flat1 = Flatten(name="flatten_brain")(pool1)

    # Second modality
    input2 = Input(shape=(1, 33), name="input_eye")
    conv2 = Conv1D(filters=32, kernel_size=1, activation='relu', name="Conv_eye")(input2)
    pool2 = MaxPooling1D(pool_size=1, name="pool_eye")(conv2)
    flat2 = Flatten(name="flatten_eye")(pool2)

    # Concatenate both branches
    merge = concatenate([flat1, flat2])

    dense1 = Dense(units=64, activation='relu')(merge)
    output = Dense(units=5, activation='softmax')(dense1)

    return Model(inputs=[input1, input2], outputs=output)
