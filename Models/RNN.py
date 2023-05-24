from keras.models import Model
from keras.layers import Dense, Input, concatenate, SimpleRNN


def create_single_modal_rnn() -> Model:
    input1 = Input(shape=(31, 10))
    rnn1 = SimpleRNN(units=64)(input1)
    output1 = Dense(units=5, activation='softmax')(rnn1)
    output = Dense(units=5, activation='softmax')(output1)

    return Model(inputs=input1, outputs=output)


def create_dual_modal_rnn() -> Model:
    # First modality
    input1 = Input(shape=(31, 10))
    rnn1 = SimpleRNN(units=64)(input1)
    output1 = Dense(units=5, activation='softmax')(rnn1)

    # Second modality
    input2 = Input(shape=(1, 33))
    rnn2 = SimpleRNN(units=64)(input2)
    output2 = Dense(units=5, activation='softmax')(rnn2)

    # Concatenate both branches
    merge = concatenate([output1, output2])

    output = Dense(units=5, activation='softmax')(merge)

    return Model(inputs=[input1, input2], outputs=output)
