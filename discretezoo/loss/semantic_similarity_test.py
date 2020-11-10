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
"""Tests for the file semantic_similarity.py"""
from absl import flags
from absl.testing import absltest
import tensorflow as tf

from discretezoo.loss import semantic_similarity


class SemanticSimilarityTest(absltest.TestCase):

  def test_cosine_distance_mean_reduction(self):
    """Tests consine similarity between two sentences."""
    # Points on the unit circle for 0, pi/6, pi/4, pi/3, and pi/2 radians.
    embeddings = tf.constant([[1., 0.], [0.866025, 0.5], [0.707107, 0.707107],
                              [0.5, 0.866025], [0., 1.]])
    # These are the points at 1.0 on the x and y axes, respectively.
    adversarial_sentences = tf.constant([[0, 4]], dtype=tf.int32)
    # This is the point at pi/4 radians on the unit circle, twice.
    original_sentences = tf.constant([[2, 2]], dtype=tf.int32)
    # The two points at 1.0 on the x and y axes added together and normalized is
    # the same point as pi/4 on the unit circle, so cosine similarity is 1.
    # The cosine distance object gives 1 - similarity, so we get 0 as a result.
    expected_cosine_distance = tf.constant([[0.0]])
    cosine_distance_object = semantic_similarity.EmbeddedCosineDistance(
        embeddings)
    test_cosine_distance = cosine_distance_object(original_sentences,
                                                  adversarial_sentences)
    tf.debugging.assert_near(expected_cosine_distance, test_cosine_distance)

  def test_cosine_distance_two_dimensional_embeddings_requirement(self):
    embeddings = tf.range(0, 10, dtype=tf.float32)
    with self.assertRaises(AssertionError):
      cosine_distance_object = semantic_similarity.EmbeddedCosineDistance(
          embeddings)

  def test_cosine_distance_output_shapes(self):
    # Create an embedding matrix with 10 vocab items and dimension 5.
    embeddings = tf.random.uniform((10, 5))

    adversarial_sentences = tf.random.uniform((4, 6), maxval=10, dtype=tf.int32)
    original_sentences = tf.random.uniform((4, 6), maxval=10, dtype=tf.int32)

    distance_object = semantic_similarity.EmbeddedCosineDistance(embeddings)

    expected_shape = (4, 1)
    distance = distance_object(original_sentences, adversarial_sentences)
    tf.debugging.assert_shapes([(distance, expected_shape)])

  def test_euclidean_distance_sum_reduction(self):
    """Tests euclidean distance between two sentences."""
    embeddings = tf.expand_dims(tf.range(0, 10, dtype=tf.float32), -1)

    adversarial_sentences = tf.constant([[0, 8]])
    original_sentences = tf.constant([[4, 4]])
    expected_distance = tf.constant([[0.]])

    distance_object = semantic_similarity.EmbeddedEuclideanDistance(
        embeddings, reduce_mean=False)
    test_distance = distance_object(original_sentences, adversarial_sentences)
    tf.debugging.assert_near(expected_distance, test_distance)

  def test_euclidean_distance_mean_reduction(self):
    """Tests euclidean distance between two sentences."""
    embeddings = tf.expand_dims(tf.range(0, 10, dtype=tf.float32), -1)

    adversarial_sentences = tf.constant([[0, 7]])
    original_sentences = tf.constant([[3, 3]])
    expected_distance = tf.constant([[0.5]])

    distance_object = semantic_similarity.EmbeddedEuclideanDistance(
        embeddings, reduce_mean=True)
    test_distance = distance_object(original_sentences, adversarial_sentences)
    tf.debugging.assert_near(expected_distance, test_distance)

  def test_euclidean_distance_two_dimensional_embeddings_requirement(self):
    embeddings = tf.range(0, 10, dtype=tf.float32)
    with self.assertRaises(AssertionError):
      distance_object = semantic_similarity.EmbeddedEuclideanDistance(
          embeddings)

  def test_euclidean_distance_output_shapes(self):
    # Create an embedding matrix with 10 vocab items and dimension 5.
    embeddings = tf.random.uniform((10, 5))

    adversarial_sentences = tf.random.uniform((4, 6), maxval=10, dtype=tf.int32)
    original_sentences = tf.random.uniform((4, 6), maxval=10, dtype=tf.int32)

    distance_object = semantic_similarity.EmbeddedEuclideanDistance(
        embeddings, reduce_mean=True)

    expected_shape = (4, 1)
    distance = distance_object(original_sentences, adversarial_sentences)
    tf.debugging.assert_shapes([(distance, expected_shape)])


if __name__ == '__main__':
  absltest.main()
