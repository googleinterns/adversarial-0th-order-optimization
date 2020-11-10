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
"""Contains the importance scorer tests."""
from absl.testing import absltest
import tensorflow as tf

from discretezoo.attack import importance


class ScorerTest(absltest.TestCase):
  """This class contains the tests for the functions inside of importance.py."""

  def test_scorer(self):
    """This tests the importance scorer as a whole."""

    def output_difference_fn(deleted_token_sentence, original_probabilities):
      return tf.cast(
          tf.math.reduce_sum(deleted_token_sentence, axis=-1, keepdims=True),
          tf.float32)

    def dropper_function(sentences, position):
      return sentences

    sentence = tf.expand_dims(tf.range(0, 10), axis=0)
    original_probabilities = tf.zeros((1, 10))
    expected_scores = tf.constant([[45.0] * 10])
    test_scores = importance.scorer(sentence,
                                    original_probabilities,
                                    output_difference_fn,
                                    dropper_function,
                                    pad_id=-1)
    tf.debugging.assert_equal(expected_scores, test_scores)

  def test_ignore_padding(self):
    """This tests whether or not padding tokens are ignored in scoring."""

    def output_difference_fn(deleted_token_sentence, original_probabilities):
      return tf.cast(
          tf.math.reduce_sum(deleted_token_sentence, axis=-1, keepdims=True),
          tf.float32)

    def dropper_function(sentences, position):
      return sentences

    sentences = tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    original_probabilities = tf.zeros((5, 5))
    expected_scores = tf.constant(
        [[5, 5, 5, 5, 5], [4, 4, 4, 4, 0], [3, 3, 3, 0, 0], [2, 2, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        dtype=tf.float32)
    test_scores = importance.scorer(sentences, original_probabilities,
                                    output_difference_fn, dropper_function)
    tf.debugging.assert_equal(expected_scores, test_scores)


class MaskPositionTest(absltest.TestCase):

  def test_mask_position(self):
    """This tests _mask_position to see if it correctly masks target tokens."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    expected_masked_position = tf.constant([[1, 2, 3, 4, 5, 0, 7, 8, 9, 10]],
                                           dtype=tf.int32)
    position = 5
    test_masked_position = importance.mask_position(sentence,
                                                    position,
                                                    mask_id=0)
    tf.debugging.assert_equal(expected_masked_position, test_masked_position)

  def test_masking_last_position(self):
    """This test is for the edge case where we mask the last token."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    expected_last_token_masked_sentence = tf.constant(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    position = 9
    test_last_token_masked_sentence = importance.mask_position(sentence,
                                                               position,
                                                               mask_id=0)
    tf.debugging.assert_equal(expected_last_token_masked_sentence,
                              test_last_token_masked_sentence)

  def test_mask_position_too_big_index(self):
    """This test is to ensure out of bounds (too big) positions get caught."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    position = 11
    with self.assertRaises(AssertionError):
      importance.mask_position(sentence, position, mask_id=0)

  def test_mask_position_negative_index(self):
    """This test ensures negative positions (while valid python) aren't used."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    position = -1
    with self.assertRaises(AssertionError):
      importance.mask_position(sentence, position, mask_id=0)

  def test_mask_position_mask_id_requirement(self):
    """This test makes sure mask_position checks for the mask_id argument."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    position = 5
    with self.assertRaises(AssertionError):
      importance.mask_position(sentence, position)


class DropPositionTest(absltest.TestCase):

  def test_drop_position(self):
    """This tests _mask_position to see if it correctly drops target tokens."""
    sentence = tf.expand_dims(tf.range(0, 10), 0)
    expected_dropped_position = tf.constant([[0, 1, 2, 3, 5, 6, 7, 8, 9]],
                                            dtype=tf.int32)
    position = 4
    test_dropped_position = importance.drop_position(sentence, position)
    tf.debugging.assert_equal(expected_dropped_position, test_dropped_position)

  def test_dropping_last_position(self):
    """This test is for the edge case where we drop the last token."""
    sentence = tf.expand_dims(tf.range(0, 10), 0)
    expected_last_token_dropped_sentence = tf.expand_dims(tf.range(0, 9), 0)
    position = 9
    test_last_token_dropped_sentence = importance.drop_position(sentence, 9)
    tf.debugging.assert_equal(expected_last_token_dropped_sentence,
                              test_last_token_dropped_sentence)

  def test_drop_position_too_big_index(self):
    """This test is to ensure out of bounds (too big) positions get caught."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    position = 11
    with self.assertRaises(AssertionError):
      importance.drop_position(sentence, position)

  def test_drop_position_negative_index(self):
    """This test ensures negative positions (while valid python) aren't used."""
    sentence = tf.expand_dims(tf.range(1, 11), 0)
    position = -1
    with self.assertRaises(AssertionError):
      importance.drop_position(sentence, position)


if __name__ == '__main__':
  absltest.main()
