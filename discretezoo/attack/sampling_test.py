"""Tests for the sampling functions in the file sampling.py"""
import tensorflow as tf

from discretezoo.attack import sampling

from absl import flags
from absl.testing import absltest


class SimilaritySamplingTest(absltest.TestCase):
  """This class contains tests for the similarity sampling functions."""

  def test_knn_similarity_cosine(self):
    """Tests that knn_similarity_cosine doesn't return input indices."""
    # Points on the unit circle for 0, pi/6, pi/4, pi/3, and pi/2 radians.
    embeddings = tf.constant([[1., 0.], [0.866025, 0.5], [0.707107, 0.707107],
                              [0.5, 0.866025], [0., 1.]])
    indices = tf.constant([[0], [4]])
    num_to_sample = 4
    expected_samples = tf.constant([[1, 2, 3, 4], [3, 2, 1, 0]], dtype=tf.int32)
    test_samples = sampling.knn_sampling_cosine(indices, embeddings,
                                                num_to_sample)
    tf.debugging.assert_equal(expected_samples, test_samples)

  def test_knn_similarity_euclidean(self):
    """Tests that knn_sampling_euclidean doesn't return input indices."""
    embeddings = tf.constant([[0], [1], [2], [3], [4]])
    indices = tf.constant([[0], [4]])
    num_to_sample = 4
    expected_samples = tf.constant([[1, 2, 3, 4], [3, 2, 1, 0]], dtype=tf.int32)
    test_samples = sampling.knn_sampling_euclidean(indices, embeddings,
                                                   num_to_sample)
    tf.debugging.assert_equal(expected_samples, test_samples)


if __name__ == '__main__':
  absltest.main()
