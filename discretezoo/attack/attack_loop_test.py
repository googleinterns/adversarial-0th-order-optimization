"""Tests for the file attack_loop.py"""
import tensorflow as tf

from absl.testing import absltest

from discretezoo.attack import estimation
from discretezoo.attack import attack_loop


class DiscreteZOOMock(estimation.DiscreteZOO):
  """A mock optimizer class that creates constant updates for target tokens.

  Attributes:
    constant_update: The value to update the target tokens with.
  """

  def __init__(self, constant_update: int):
    """Initializes the mock optimizer with the constant update value.

    Arguments:
      constant_update: The value to update the target tokens with.
    """
    self._constant_update = constant_update
    return

  def replace_token(self, sentences: tf.Tensor, original_sentences: tf.Tensor,
                    labels: tf.Tensor, indices: tf.Tensor,
                    iterations: int) -> tf.Tensor:
    """This function mocks replace_token in DiscreteZOO by returning a constant.

    With the exception of sentences, all arguments to this function are ignored.
    Sentences is only used to get the batch size for the update we return.

    Arguments:
      sentences: Sentences is used to calculate the batch size for the update.
      original_sentences: The original sentences before any tokens are updated.
        Not used here.
      labels: These are the labels used in the adversarial loss function.
        Not used here.
      indices: Indices is the location of the target token in sentences.
        Not used here.
      iterations: This controls how many times a single token can be updated.
        Not used here.

    Returns:
      A tensor <int32>[batch_size, 1] filled with constant_update.
    """
    batch_size = sentences.shape[0]
    update = tf.constant([[self._constant_update]] * batch_size)
    return update


class AttackLoopTest(absltest.TestCase):
  """This class contains the tests for the function loop in attack_loop.py"""

  def test_attack_loop(self):
    """Tests that sentences aren't updated further once finished."""

    def count_updated_tokens(adversarial_sentences: tf.Tensor,
                             labels: tf.Tensor) -> tf.Tensor:
      target_counts = tf.reshape(tf.range(9, -1, -1), (10, 1))
      current_counts = tf.reduce_sum(adversarial_sentences,
                                     axis=-1,
                                     keepdims=True)
      return target_counts == current_counts

    optimizer = DiscreteZOOMock(0)
    sentences = tf.ones((10, 10), dtype=tf.int32)
    # Labels are required for the attack_loop but ignored by DiscreteZOOMock.
    labels = tf.zeros((10, 1), dtype=tf.int32)
    # This sets index 0 to the most importance token and index 9 to the least.
    importance_scores = tf.stack([tf.range(10, 0, -1)] * 10, axis=0)
    iterations_per_token = 1
    max_changes = 10
    test_adversarial_sentences, test_finished_attacks = attack_loop.loop(
        sentences,
        labels,
        optimizer,
        importance_scores,
        count_updated_tokens,
        iterations_per_token,
        max_changes,
        padding_index=2)
    # Because of the stopping values picked, we will have 9 ones in the first
    # sentence, 8 ones in the next, and so on. This is an upper triangular
    # matrix with the diagonal set to 0s.
    expected_adversarial_sentences = tf.constant(
        [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    expected_finished_attacks = tf.ones((10, 1), dtype=tf.bool)

    tf.debugging.assert_equal(expected_adversarial_sentences,
                              test_adversarial_sentences)

    tf.debugging.assert_equal(expected_finished_attacks, test_finished_attacks)

  def test_attack_loop_failed_attack(self):
    """Tests that the adversarial sentence is returned for failed attacks."""

    def count_updated_tokens(adversarial_sentences: tf.Tensor,
                             labels: tf.Tensor) -> tf.Tensor:
      target_counts = tf.constant([[11], [1]], dtype=tf.int32)
      current_counts = tf.reduce_sum(adversarial_sentences,
                                     axis=-1,
                                     keepdims=True)
      return target_counts == current_counts

    optimizer = DiscreteZOOMock(0)
    sentences = tf.ones((2, 10), dtype=tf.int32)
    # Labels are required for the attack_loop but ignored by DiscreteZOOMock.
    labels = tf.zeros((2, 1), dtype=tf.int32)
    # This sets index 0 to the most importance token and index 9 to the least.
    importance_scores = tf.stack([tf.range(10, 0, -1)] * 2, axis=0)
    iterations_per_token = 1
    max_changes = 10
    test_adversarial_sentences, test_finished_attacks = attack_loop.loop(
        sentences,
        labels,
        optimizer,
        importance_scores,
        count_updated_tokens,
        iterations_per_token,
        max_changes,
        padding_index=2)
    # The first sentence is the unsuccessful adversarial attack, where all
    # tokens are set to 0s and the second sentence is the successful attack
    # where all tokens but the last are 0s.
    expected_adversarial_sentences = tf.constant(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    expected_finished_attacks = tf.constant([[False], [True]])

    tf.debugging.assert_equal(expected_adversarial_sentences,
                              test_adversarial_sentences)

    tf.debugging.assert_equal(expected_finished_attacks, test_finished_attacks)

  def test_attack_loop_padded_sentences(self):
    """Test that padding tokens aren't able to be changed in the attack."""

    changed_token_id = 2

    def count_updated_tokens(adversarial_sentences: tf.Tensor,
                             labels: tf.Tensor) -> tf.Tensor:
      target_counts = tf.constant([[10], [10]], dtype=tf.int32)
      updated_tokens = tf.cast((adversarial_sentences == changed_token_id),
                               tf.int32)
      current_counts = tf.reduce_sum(updated_tokens, axis=-1, keepdims=True)
      return target_counts == current_counts

    optimizer = DiscreteZOOMock(changed_token_id)
    sentences = tf.constant(
        [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=tf.int32)
    labels = tf.zeros((2, 1), dtype=tf.int32)
    # This sets index 0 to the most importance token and index 9 to the least.
    importance_scores = tf.stack([tf.range(10, 0, -1)] * 2, axis=0)
    iterations_per_token = 1
    max_changes = 10

    test_adversarial_sentences, test_finished_attacks = attack_loop.loop(
        sentences, labels, optimizer, importance_scores, count_updated_tokens,
        iterations_per_token, max_changes)

    expected_adversarial_sentences = tf.constant(
        [[2, 2, 2, 2, 2, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        dtype=tf.int32)

    tf.debugging.assert_equal(test_adversarial_sentences,
                              expected_adversarial_sentences)


if __name__ == '__main__':
  absltest.main()
