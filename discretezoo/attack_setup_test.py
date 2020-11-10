"""
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Tests for the file discretezoo_main.py"""
from absl.testing import absltest
import tensorflow as tf

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
