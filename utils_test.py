import unittest

import numpy as np

from utils import Utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.util = Utils()
        self.window_size = 3
        self.sentence = "Er sieht.\nIvana gibt.\nStefan mag.\nStefan am 3. Mai"
        self.data_set_char_gold = [(1.0, "eht. Iv"), (1.0, "ibt. St"),
                                   (1.0, "mag. St"), (0.0, 'm 3. Ma')]

        self.char_2_id_dict_gold = \
            {'e': 1, 'h': 2, 't': 3, '.': 4, ' ': 5, 'I': 6, 'v': 7, 'i': 8,
             'b': 9, 'S': 10, 'm': 11, 'a': 12, 'g': 13, '3': 14, 'M': 15}

        self.data_set_gold = \
            [(1.0, np.array([float(i) for i in [1, 2, 3, 4, 5, 6, 7]])),
             (1.0, np.array([float(i) for i in [8, 9, 3, 4, 5, 10, 3]])),
             (1.0, np.array([float(i) for i in [11, 12, 13, 4, 5, 10, 3]])),
             (0.0, np.array([float(i) for i in [11, 5, 14, 4, 5, 15, 12]]))
             ]

        self.test_sentence = "Er sagt: ›Das ist so?‹ Wir machen “weiter”! Oki haben! – Aber dann\n"
        self.potential_eos_list_gold = [(7, "agt: ›D"), (20, " so?‹ W"),
                                        (42, "er”! Ok"),
                                        (53, "ben! – ")]

    def test_build_char_2_id_dict(self):
        char_2_id_dict_cur = \
            self.util.build_char_2_id_dict(self.data_set_char_gold, 1)

        self.assertEqual(
            len(char_2_id_dict_cur), len(
                self.char_2_id_dict_gold))
        self.assertDictEqual(char_2_id_dict_cur, self.char_2_id_dict_gold)

    def test_build_data_set(self):
        data_set_cur = self.util.build_data_set(self.data_set_char_gold,
                                                self.char_2_id_dict_gold,
                                                self.window_size)

        self.assertEqual(len(data_set_cur), len(self.data_set_gold))

        for i in range(0, len(self.data_set_gold)):
            feature_vector_equals = np.array_equal(data_set_cur[i][1],
                                                   self.data_set_gold[i][1])
            self.assertEqual(data_set_cur[i][0], self.data_set_gold[i][0])
            self.assertEqual(feature_vector_equals, True)

    def test_build_data_set_char(self):
        data_set_char_cur = self.util.build_data_set_char(self.sentence,
                                                          self.window_size)

        self.assertEqual(data_set_char_cur, self.data_set_char_gold)

    def test_build_potential_eos_list(self):
        potential_eos_list_cur = self.util.build_potential_eos_list(
            self.test_sentence, self.window_size)

        self.assertEqual(len(potential_eos_list_cur),
                         len(self.potential_eos_list_gold))

        for i in range(0, len(self.potential_eos_list_gold)):
            self.assertEqual(potential_eos_list_cur[i],
                             self.potential_eos_list_gold[i])


if __name__ == '__main__':
    unittest.main()
