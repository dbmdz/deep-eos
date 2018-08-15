"""Implement a sentence boundary detection system (deep-eos)."""
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.noise import AlphaDropout
from keras.models import Sequential
from keras.models import load_model
from keras.utils import plot_model

from utils import Utils


class EOS():
    """Implementation of the end-of-sentence classification logic."""

    def __init__(self):
        """Implement the EOS constructor."""
        self.util = Utils()

    def train(self, training_file,  # pylint: disable-msg=too-many-arguments,too-many-locals,too-many-statements # noqa: E501
              architecture="cnn",
              window_size=4,
              epochs=5,
              batch_size=32,
              dropout=0.25,
              min_freq=10000,
              max_features=20000,
              embedding_size=128,
              lstm_gru_size=256,
              mlp_dense=6,
              mlp_dense_units=16,
              kernel_size=5,
              filters=64,
              hidden_dims=250,
              strides=1,
              model_filename='best_model.hdf5',
              vocab_filename='vocab.dump'):
        """Train a classifier for end-of-sentence detection."""
        with open(training_file, mode='r', encoding='utf-8') as f_p:
            training_corpus = f_p.read()

        data_set_char = self.util.build_data_set_char(
            training_corpus, window_size)
        char_2_id_dict = self.util.build_char_2_id_dict(
            data_set_char, min_freq)

        data_set = self.util.build_data_set(data_set_char, char_2_id_dict,
                                            window_size)

        x_train = np.array([i[1] for i in data_set])
        y_train = np.array([i[0] for i in data_set])

        maxlen = 2 * window_size + 1

        model = Sequential()

        if architecture == "cnn":
            model.add(Embedding(max_features,
                                embedding_size,
                                input_length=maxlen))
            model.add(Dropout(dropout))

            # we add a Convolution1D, which will learn filters
            # word group filters of size filter_length:
            model.add(Conv1D(filters,
                             kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=strides))
            # we use max pooling:
            model.add(GlobalMaxPooling1D())

            # We add a vanilla hidden layer:
            model.add(Dense(hidden_dims))
            model.add(Dropout(dropout))
            model.add(Activation('relu'))

            # We project onto a single unit output layer, and squash it with a
            # sigmoid:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

        elif architecture == "lstm":
            model.add(Embedding(max_features,
                                embedding_size))

            model.add(
                LSTM(
                    lstm_gru_size,
                    dropout=dropout,
                    recurrent_dropout=dropout))

            model.add(Dense(1, activation='sigmoid'))

        elif architecture == "bi-lstm":
            model.add(Embedding(max_features,
                                embedding_size))

            model.add(
                Bidirectional(
                    LSTM(
                        lstm_gru_size,
                        dropout=dropout,
                        recurrent_dropout=dropout)))

            model.add(Dense(1, activation='sigmoid'))

        elif architecture == "gru":
            model.add(Embedding(max_features,
                                embedding_size))

            model.add(GRU(lstm_gru_size, dropout=dropout,
                          recurrent_dropout=dropout))

            model.add(Dense(1, activation='sigmoid'))

        elif architecture == "bi-gru":
            model.add(Embedding(max_features,
                                embedding_size))

            model.add(Bidirectional(
                GRU(lstm_gru_size, dropout=dropout, recurrent_dropout=dropout))) # noqa: E501

            model.add(Dense(1, activation='sigmoid'))

        elif architecture == "mlp":
            model.add(Dense(mlp_dense_units, input_shape=(maxlen,),
                            kernel_initializer='lecun_normal'))
            model.add(Activation('selu'))
            model.add(AlphaDropout(dropout))

            for _ in range(mlp_dense - 1):
                model.add(
                    Dense(
                        mlp_dense_units,
                        kernel_initializer='lecun_normal'))
                model.add(Activation('selu'))
                model.add(AlphaDropout(dropout))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        plot_model(model, to_file='model.png')

        mcp = ModelCheckpoint(model_filename,
                              monitor="acc",
                              save_best_only=True, save_weights_only=False,
                              mode='max')

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  callbacks=[mcp])

        self.util.save_vocab(char_2_id_dict, vocab_filename)

    def test(self, test_file,  # pylint: disable-msg=too-many-arguments,too-many-locals # noqa: E501
             model_filename='best_model.hdf5',
             vocab_filename='vocab.dump',
             window_size=4,
             batch_size=32):
        """Test a classifier for end-of-sentence detection."""
        with open(test_file, mode='r', encoding='utf-8') as f_p:
            test_corpus = f_p.read()

        char_2_id_dict = self.util.load_vocab(vocab_filename)

        data_set_char = self.util.build_data_set_char(test_corpus, window_size)

        data_set = self.util.build_data_set(data_set_char, char_2_id_dict,
                                            window_size)

        x_test = np.array([i[1] for i in data_set])
        y_test = np.array([i[0] for i in data_set])

        model = load_model(model_filename)

        scores = model.evaluate(x_test, y_test, batch_size=batch_size)

        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    def tag(self, input_file,  # pylint: disable-msg=too-many-arguments,too-many-locals # noqa: E501
            model_filename='best_model.hdf5',
            vocab_filename='vocab.dump',
            window_size=4,
            batch_size=32,
            eos_marker="</eos>"):
        """Tag an input file with end-of-sentence markers."""
        char_2_id_dict = self.util.load_vocab(vocab_filename)

        model = load_model(model_filename)

        with open(input_file, mode='r', encoding='utf-8') as f_p:
            t = f_p.read()  # pylint: disable-msg=invalid-name

        potential_eos_list = self.util.build_potential_eos_list(t, window_size)

        eos_counter = 0

        for potential_eos in potential_eos_list:
            start, char_sequence = potential_eos

            data_set = self.util.build_data_set([(-1.0, char_sequence)],
                                                char_2_id_dict,
                                                window_size)

            if data_set:
                _, feature_vector = data_set[0]

                predicted = model.predict(
                    feature_vector.reshape(
                        1,
                        2 * window_size + 1),
                    batch_size=batch_size,
                    verbose=0)

                if predicted[0][0] >= 0.5:
                    t = t[:(eos_counter * len(eos_marker)) + start + 1] + eos_marker + t[(eos_counter * len(eos_marker)) + start + 1:]  # pylint: disable-msg=invalid-name # noqa: E501
                    eos_counter += 1

        print(t[:] + eos_marker)

    def extract(self, input_file,
                window_size=4):
        """Extract tabulator separated list of character window and label."""
        with open(input_file, mode='r', encoding='utf-8') as f_p:
            input_corpus = f_p.read()

        data_set_char = self.util.build_data_set_char(
            input_corpus, window_size)

        print("\n".join([str(int(entry[0])) + "\t" + "".join([str(id_) for id_ in entry[1]]) for entry in data_set_char]))  # noqa: E501
