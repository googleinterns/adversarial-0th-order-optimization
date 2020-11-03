"""Tests for the file estimation.py"""
from absl import flags
from absl.testing import absltest
import tensorflow as tf

from discretezoo.attack import estimation


class EstimationTest(absltest.TestCase):
  """This class contains test cases for the file estimation.py"""

  def test_estimation(self):
    """Tests the estimation class on embeddings that are cube vertices.

    We want to see if we can predict token indices that we have not queried the
    model with yet. In this test we are querying the "model" with two indices
    that correspond to the vectors [0., 0., 1.] and [1., 1., 0.]. If the class
    functions correctly, it should predict the index that corresponds to the
    vector [1., 1., 1.] because that is the vector that maximizes our loss.
    """
    test_embeddings = tf.constant([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                   [1., 0., 0.], [1., 1., 0.], [0., 1., 1.],
                                   [1., 0., 1.], [1., 1., 1.]])
    sentence = tf.constant([[0], [7]], dtype=tf.int32)
    position = tf.constant([[0], [0]], dtype=tf.int32)
    original_sentence = sentence
    # Labels are required by DiscreteZOO but unused by our loss function.
    labels = tf.constant([[0], [0]], dtype=tf.int32)
    # We expect the index 7, because it corresponds to the highest-loss vector.
    expected_replacement = tf.constant([[7], [0]])

    # pylint: disable=unused-argument
    def sampler(indices, embeddings, num_to_sample):
      # These are the ids of the tokens that we will "sample".
      return tf.constant([[4, 1], [4, 1]])

    def adversarial_loss(original_sentence, labels, replacement_sentence):
      """Our loss function is the l2 difference between the two sentences."""
      original = tf.nn.embedding_lookup(test_embeddings, original_sentence)
      replacement = tf.nn.embedding_lookup(test_embeddings,
                                           replacement_sentence)
      loss = tf.expand_dims(tf.norm(original - replacement, axis=[1, 2]), -1)
      return loss

    # Create our DiscreteZoo object to perform gradient ascent to maximize loss.
    test_discrete_zoo = estimation.DiscreteZOO(sampler,
                                               test_embeddings,
                                               adversarial_loss,
                                               num_to_sample=2,
                                               reduce_mean=False,
                                               descent=False)

    replacement_token = test_discrete_zoo.replace_token(sentence,
                                                        original_sentence,
                                                        labels, position, 1)
    tf.debugging.assert_equal(replacement_token, expected_replacement)

  def test_estimation_with_added_displacement(self):
    """Tests the estimation class on embeddings that are cube vertices.

    This is similar to the previous test, except that now, we add the embedding
    and displacement together and then calculate similarity. We still try to
    get to the same predicted tokens, but since we now take into account the
    starting position, we need to change the tokens in the original_sentences.
    """
    test_embeddings = tf.constant([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                   [1., 0., 0.], [1., 1., 0.], [0., 1., 1.],
                                   [1., 0., 1.], [1., 1., 1.]])
    sentence = tf.constant([[1], [3]], dtype=tf.int32)
    position = tf.constant([[0], [0]], dtype=tf.int32)
    original_sentence = sentence
    # Labels are required by DiscreteZOO but unused by our loss function.
    labels = tf.constant([[0], [0]], dtype=tf.int32)
    # We expect the index 7, because it corresponds to the highest-loss vector.
    expected_replacement = tf.constant([[7], [7]])

    # pylint: disable=unused-argument
    def sampler(indices, embeddings, num_to_sample):
      # These are the ids of the tokens that we will "sample".
      return tf.constant([[5, 6], [4, 6]])

    def adversarial_loss(original_sentence, labels, replacement_sentence):
      """Our loss function is the l2 difference between the two sentences."""
      original = tf.nn.embedding_lookup(test_embeddings, original_sentence)
      replacement = tf.nn.embedding_lookup(test_embeddings,
                                           replacement_sentence)
      loss = tf.expand_dims(tf.norm(original - replacement, axis=[1, 2]), -1)
      return loss

    # Create our DiscreteZoo object to perform gradient ascent to maximize loss.
    test_discrete_zoo = estimation.DiscreteZOO(
        sampler,
        test_embeddings,
        adversarial_loss,
        num_to_sample=2,
        reduce_mean=False,
        descent=False,
        add_displacement_to_embedding=True,
        discretize_by_cosine=True)

    replacement_token = test_discrete_zoo.replace_token(sentence,
                                                        original_sentence,
                                                        labels, position, 1)
    tf.debugging.assert_equal(replacement_token, expected_replacement)

  def test_scatter_helper(self):
    sentences = tf.zeros((10, 10))
    # Our scatter_helper function expects a shape of [batch_size, 1]
    indices = tf.expand_dims(tf.range(0, 10, dtype=tf.int32), -1)
    # DiscreteZoo requires the shape [batch_size, 1] but is being bypassed here.
    updates = tf.ones((10,))
    # This creates a matrix with ones on the main diagonal and zeros elsewhere.
    updated_sentences = tf.linalg.diag(updates)

    test_updated_sentences = estimation.DiscreteZOO.scatter_helper(
        sentences, indices, updates)
    tf.debugging.assert_equal(test_updated_sentences, updated_sentences)


if __name__ == '__main__':
  absltest.main()
