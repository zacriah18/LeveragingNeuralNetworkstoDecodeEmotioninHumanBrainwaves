from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling1D, Dropout, Dense, MultiHeadAttention, LayerNormalization, Add, Concatenate, Input


def transformer_block(inputs, num_heads, dim, dropout_rate, prefix):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout_rate, name=f"{prefix}_mha")(inputs, inputs)
    out1 = Add(name=f"{prefix}_add_1")([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_1")(out1)

    ffn_output = Dense(dim, activation="relu", name=f"{prefix}_dense_1")(out1)
    ffn_output = Dense(inputs.shape[-1], activation="relu", name=f"{prefix}_dense_2")(ffn_output)
    out2 = Add(name=f"{prefix}_add_2")([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_2")(out2)
    return out2


def create_single_modal_Transformer() -> Model:
    inputs = Input(shape=(31, 10))
    x = Dense(32, activation="relu")(inputs)
    x = transformer_block(x, num_heads=2, dim=64, dropout_rate=0.1, prefix="brainonly")
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(5, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)


def create_dual_modal_Transformer() -> Model:
    input_brainwave = Input(shape=(31, 10))
    x_brainwave = Dense(32, activation="relu", name="dense_brain_1")(input_brainwave)
    x_brainwave = transformer_block(x_brainwave, num_heads=4, dim=64, dropout_rate=0.1, prefix="brain")
    x_brainwave = GlobalAveragePooling1D(name="average_brain")(x_brainwave)

    input_eye = Input(shape=(1, 33))
    x_eye = Dense(32, activation="relu", name="dense_eye_1")(input_eye)
    x_eye = transformer_block(x_eye, num_heads=4, dim=64, dropout_rate=0.1, prefix="eye")
    x_eye = GlobalAveragePooling1D(name="average_eye")(x_eye)

    # Combine the brainwave and eye movement features
    combined = Concatenate()([x_brainwave, x_eye])

    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(combined)
    x = Dropout(0.1)(x)
    output = Dense(5, activation="softmax")(x)

    return Model(inputs=[input_brainwave, input_eye], outputs=output)
