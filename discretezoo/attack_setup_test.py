"""Tests for the file discretezoo_main.py"""
import tensorflow as tf

from absl.testing import absltest
from discretezoo import attack_setup


class SortDatasetTest(absltest.TestCase):
  """This class tests the utility function sort_dataset in attack_setup"""

  def test_dataset_sorting(self):
    dataset_examples = {
        'sentence':
            tf.ragged.constant(['a b c d e', 'a b', 'a', 'a b c', 'a b c d']),
        'index':
            tf.constant([5, 2, 1, 3, 4], dtype=tf.int32)
    }
    expected_dataset_examples = {
        'sentence':
            tf.ragged.constant([['a'], ['a', 'b'], ['a', 'b', 'c'],
                                ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd',
                                                       'e']]),
        'index':
            tf.constant([1, 2, 3, 4, 5], dtype=tf.int32),
        'token_count':
            tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
    }

    expected_dataset = tf.data.Dataset.from_tensor_slices(
        expected_dataset_examples)
    input_dataset = tf.data.Dataset.from_tensor_slices(dataset_examples)

    test_dataset = attack_setup.sort_dataset(input_dataset)
    for expected_example, test_example in zip(expected_dataset, test_dataset):
      self.assertSetEqual(set(expected_example.keys()),
                          set(test_example.keys()))
      for key in expected_example.keys():
        tf.debugging.assert_equal(expected_example[key], test_example[key])


if __name__ == '__main__':
  absltest.main()
