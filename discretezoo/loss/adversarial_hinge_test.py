"""Tests for the file adversarial_hinge.py"""
import tensorflow as tf

from absl import flags
from absl.testing import absltest

from discretezoo.loss import adversarial_hinge


class UntargetedLossTest(absltest.TestCase):
  """Tests for the untargeted_loss adversarial loss function."""

  def test_untargeted_loss(self):
    """Tests that the function returns expected values on a non-edge case."""
    true_probability = 0.8
    true_label = 0
    max_other_label = 2
    probability_list = [0.0] * 10
    probability_list[true_label] = true_probability
    probability_list[max_other_label] = 1.0 - true_probability
    probability_vector = tf.constant([probability_list])
    true_label_vector = tf.constant([[true_label]], dtype=tf.int32)
    # This should return log(0.8) - log(0.2) = log(0.8/0.2)
    test_loss = adversarial_hinge.untargeted_loss(probability_vector,
                                                  true_label_vector, 0.0)
    expected_loss = tf.math.log(
        tf.constant(true_probability / (1 - true_probability)))
    tf.debugging.assert_near(test_loss, expected_loss)

  def test_untargeted_loss_uniform_distribution(self):
    """Test the edge case where the model predicts the uniform distribution."""
    true_probability = 0.1
    true_label = 4
    max_other_label = 5
    probability_list = [0.1] * 10
    probability_vector = tf.constant([probability_list])
    true_label_vector = tf.constant([[true_label]], dtype=tf.int32)

    # This should return log(0.1) - log(0.1) = log(0.1/0.1) = log(1) = 0
    test_loss = adversarial_hinge.untargeted_loss(probability_vector,
                                                  true_label_vector, 0.0)
    expected_loss = tf.constant([0.0])
    tf.debugging.assert_near(test_loss, expected_loss)

  def test_untargeted_loss_nonzero_kappa(self):
    """Test edge case where model output is uniform and kappa is nonzero."""
    true_probability = 0.1
    true_label = 4
    max_other_label = 5
    kappa = 0.1
    probability_list = [0.1] * 10
    probability_vector = tf.constant([probability_list])
    true_label_vector = tf.constant([[true_label]], dtype=tf.int32)

    # This should return log(0.1) - log(0.1) = log(0.1/0.1) = log(1) = 0
    test_loss = adversarial_hinge.untargeted_loss(probability_vector,
                                                  true_label_vector, kappa)

    tf.debugging.assert_near(test_loss, tf.constant([kappa]))

  def test_untargeted_loss_overconfident_model(self):
    """Test the edge case where the model predicts 1.0 for the true class."""
    true_label = 8
    max_other_label = 0
    true_probability = 1.0
    probability_list = [0.0] * 10
    probability_list[true_label] = true_probability
    probability_vector = tf.constant([probability_list])
    true_label_vector = tf.constant([[true_label]], dtype=tf.int32)

    # This should return log(1.0) - log(0.0) = 0.0 - (-inf) = +inf
    test_loss = adversarial_hinge.untargeted_loss(probability_vector,
                                                  true_label_vector, 0.0)

    tf.debugging.assert_equal(test_loss, tf.constant([float('inf')]))


class TargetedLossTest(absltest.TestCase):
  """Tests for the targeted_loss adversarial loss function."""

  def test_targeted_loss(self):
    """Test for the standard case where the attack was not yet successful."""
    target_probability = 0.3
    target_label = 0
    max_other_label = 2
    probability_list = [0.0] * 10
    probability_list[target_label] = target_probability
    probability_list[max_other_label] = 1.0 - target_probability
    probability_vector = tf.constant([probability_list])
    target_label_vector = tf.constant([target_label], dtype=tf.int32)

    # This should return log(0.7) - log(0.3) = log(0.7/0.3).
    test_loss = adversarial_hinge.targeted_loss(probability_vector,
                                                target_label_vector, 0.0)
    expected_loss = tf.math.log(
        tf.constant((1 - target_probability) / target_probability))
    tf.debugging.assert_near(test_loss, expected_loss)

  def test_targeted_loss_uniform_distribution(self):
    """Test edge case where the model predicts the uniform distribution."""
    target_probability = 0.1
    target_label = 0
    max_other_label = 2
    probability_list = [0.1] * 10
    probability_vector = tf.constant([probability_list])
    target_label_vector = tf.constant([target_label], dtype=tf.int32)

    # It should return log(0.1) - log(0.1) = log(0.1/0.1) = log(1) = 0.0.
    test_loss = adversarial_hinge.targeted_loss(probability_vector,
                                                target_label_vector, 0.0)

    tf.debugging.assert_near(test_loss, 0.0)

  def test_targeted_loss_nonzero_kappa(self):
    """Test edge case where it's the uniform distribution and kappa > 0.0."""
    target_probability = 0.3
    target_label = 0
    max_other_label = 2
    kappa = 0.1
    probability_list = [0.1] * 10
    probability_vector = tf.constant([probability_list])
    target_label_vector = tf.constant([target_label], dtype=tf.int32)

    # It should return log(0.1) - log(0.1) + kappa = kappa
    test_loss = adversarial_hinge.targeted_loss(probability_vector,
                                                target_label_vector, kappa)

    tf.debugging.assert_near(test_loss, tf.constant([kappa]))

  def test_targeted_loss_overconfident_model(self):
    """Test the case where the model is overconfident about its prediction."""
    target_label = 8
    original_label = 0
    probability_list = [0.0] * 10
    probability_list[original_label] = 1.0
    probability_vector = tf.constant([probability_list])
    target_label_vector = tf.constant([target_label], dtype=tf.int32)

    # This should return log(1.0) - log(0.0) = 0.0 - (-inf) = +inf.
    test_loss = adversarial_hinge.targeted_loss(probability_vector,
                                                target_label_vector, 0.0)

    tf.debugging.assert_equal(test_loss, tf.constant([float('inf')]))


if __name__ == '__main__':
  absltest.main()
