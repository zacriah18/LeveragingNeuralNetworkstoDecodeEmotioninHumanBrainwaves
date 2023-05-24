from Models import utility_functions, CNN, RNN, LSTM, Transformer

def main():
    x_train_brainwaves, x_test_brainwaves, y_train_brainwaves, y_test_brainwaves = utility_functions.load_data()

    # changed parameters to specify eye movement dataset instead

    x_train_eye, x_test_eye, y_train_eye, y_test_eye = utility_functions.load_data(True, num_features=33)

    # Single Modal

    models = {
        "CNN": [CNN.create_single_modal_CNN(),
                CNN.create_dual_modal_CNN(),
                CNN.create_single_modal_CNN()],

        "RNN":[RNN.create_single_modal_rnn(),
               RNN.create_dual_modal_rnn(),
               RNN.create_single_modal_rnn()],

        "LSTM":[LSTM.create_single_modal_LSTM(),
                LSTM.create_dual_modal_LSTM(),
                LSTM.create_single_modal_LSTM()],

        "Transformer": [Transformer.create_single_modal_Transformer(),
                        Transformer.create_dual_modal_Transformer(),
                        Transformer.create_single_modal_Transformer()]
    }

    for model_set in models:
        for model in model_set:
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        single_modal = model_set[0]
        single_modal.fit(x_train_brainwaves, y_train_brainwaves, validation_split=0.2, epochs=100, batch_size=32)
        single_modal.evaluate(x_test_brainwaves, y_test_brainwaves, verbose=0)

        dual_modal = model_set[1]
        dual_modal.fit([x_train_brainwaves, x_train_eye], y_train_brainwaves, validation_split=0.2, epochs=100, batch_size=32)
        dual_modal.evaluate([x_test_brainwaves, x_test_eye], y_test_brainwaves, verbose=0)

        single_modal_dual_weights = model_set[2]
        utility_functions.transfer_weights(transfer_from=dual_modal, transfer_to=single_modal_dual_weights)
        single_modal_dual_weights.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        single_modal_dual_weights.evaluate(x_test_brainwaves, y_test_brainwaves, verbose=0)

if __name__ == "__main__":
    main()