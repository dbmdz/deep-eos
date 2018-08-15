import pickle
import re
from collections import defaultdict

import numpy as np


class Utils(object):
    def build_char_2_id_dict(self, data_set_char, min_freq):
        """ Builds a char_to_id dictionary

        This methods builds a frequency list of all chars in the data set.

        Then every char gets an own and unique index. Notice: the 0 is reserved
        for unknown chars later, so id labelling starts at 1.

        # Arguments
          data_set_char: The input data set (consisting of char sequences)
          min_freq     : Defines the minimum frequecy a char must appear in data set

        # Returns
          char_2_id dictionary
        """
        char_freq = defaultdict(int)
        char_2_id_table = {}

        for char in [char for label, seq in data_set_char for char in seq]:
            char_freq[char] += 1

        id_counter = 1

        for k, v in [(k, v) for k, v in char_freq.items() if v >= min_freq]:
            char_2_id_table[k] = id_counter
            id_counter += 1

        return char_2_id_table

    def build_data_set(self, data_set_char, char_2_id_dict, window_size):
        """ Builds a "real" data set with numpy compatible feature vectors

        This method converts the data_set_char to real numpy compatible feature
        vectors. It does also length checks of incoming and outgoing feature
        vectors to make sure that the exact window size is kept

        # Arguments
          data_set_char : The input data set (consisting of char sequences)
          char_2_id_dict: The char_to_id dictionary
          window_size   : The window size for the current model

        # Returns
          A data set which contains numpy compatible feature vectors
        """

        data_set = []

        for label, char_sequence in data_set_char:
            ids = []

            if len(char_sequence) == 2 * window_size + 1:
                for char in char_sequence:
                    if char in char_2_id_dict:
                        ids.append(char_2_id_dict[char])
                    else:
                        ids.append(0)

                feature_vector = np.array([float(ids[i])
                                           for i in range(0, len(ids))], dtype=float)

                data_set.append((float(label), feature_vector))

        return data_set

    def build_data_set_char(self, t, window_size):
        """ Builds data set from corpus

        This method builds a dataset from the training corpus

        # Arguments
          t          : Input text
          window_size: The window size for the current model

        # Returns
          A data set which contains char sequences as feature vectors
        """

        data_set_char_eos = \
            [(1.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer('[\.:?!;][^\n]?[\n]', t)]

        data_set_char_neos = \
            [(0.0, t[m.start() - window_size:m.start()].replace("\n", " ") +
              t[m.start():m.start() + window_size + 1].replace("\n", " "))
             for m in re.finditer('[\.:?!;][^\s]?[ ]+', t)]

        return data_set_char_eos + data_set_char_neos

    def build_potential_eos_list(self, t, window_size):
        """ Builds a list of potential eos from a given text

        This method builds a list of potential end-of-sentence positions from
        a given text.

        # Arguments
          t          : Input text
          window_size: The window size for the current model

        # Returns
          A list of a pair, like:
            [(1.0, "eht Iv")]
          So the first position in the pair indicates the start position for a
          potential eos. The second position holds the extracted character sequence.
        """

        PUNCT = '[\(\)\u0093\u0094`“”\"›〈⟨〈<‹»«‘’–\'``'']*'
        EOS = '([\.:?!;])'

        eos_positions = [(m.start())
                         for m in re.finditer(r'([\.:?!;])(\s+' + PUNCT + '|' +
                                              PUNCT + '\s+|[\s\n]+)', t)]

        # Lets extract 2* window_size before and after eos position and remove
        # punctuation

        potential_eos_position = []

        for eos_position in eos_positions:
            left_context = t[eos_position - (2 * window_size):eos_position]
            right_context = t[eos_position:eos_position + (3 * window_size)]

            cleaned_left_context = left_context
            cleaned_right_context = right_context

            # cleaned_left_context = re.sub(PUNCT, '', left_context)
            # cleaned_right_context = re.sub(PUNCT, '', right_context)

            # Also replace multiple whitespaces (use *only* one whitespace)
            cleaned_left_context = re.sub('\s+', ' ', cleaned_left_context)
            cleaned_right_context = re.sub('\s+', ' ', cleaned_right_context)

            potential_eos_position.append((eos_position,
                                           cleaned_left_context[-window_size:] + t[eos_position] +
                                           cleaned_right_context[1:window_size + 1]))

        return potential_eos_position

    def save_vocab(self, char_2_id_dict, vocab_filename):
        """ Saves vocabulary to a file

        # Arguments
          char_2_id_dict: The char_to_id dictionary
          vocab_filename: The output filename
        """
        with open(vocab_filename, 'wb') as f:
            pickle.dump(char_2_id_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_vocab(self, vocab_filename):
        """ Loads vocabulary from file

        # Arguments
          vocab_filename: The vocabulary filename to be read in

        # Returns
          Dictionary of vocabulary
        """
        with open(vocab_filename, 'rb') as f:
            return pickle.load(f)
