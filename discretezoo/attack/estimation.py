"""Estimation contains a class that estimates gradients in discrete space."""

import tensorflow as tf
from typing import Callable


class DiscreteZOO:
  """Contains the logic to flip discrete tokens in a sentence to optimize loss.

  This class estimates gradients in discrete space by mapping discrete tokens
  to an embedding space. We can perturb embeddings in this space by sampling
  replacement tokens and using the displacement vector between the tokens'
  embeddings. Norming the displacement vector and multiplying by the difference
  in loss yields a vector which points in the direction of increasing loss.

  Attributes:
    sampling_strategy: A callable that samples replacement tokens to test.
    embeddings: The embeddings that we are using to guide the search.
    adversarial_loss: A callable that evaluates the adversarial loss;
      takes both the original sentence and current sentence as arguments. Output
      of this function should be a tensor <float32>[batch_size, 1].
    num_to_sample: A hyperparameter that controls how many tokens we sample.
    reduce_mean: A boolean that flags if gradients are reduced with mean.
      If this is false then we will use reduce_sum.
    descent: Boolean flag to determine if we're increasing or decreasing loss.
    norm_embeddings: Boolean flag that determines if embeddings should be normed
      for the discretization step.
  """

  def __init__(self,
               sampling_strategy: Callable[[tf.Tensor, tf.Tensor, int],
                                           tf.Tensor],
               embeddings: tf.Tensor,
               adversarial_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
               num_to_sample: int = 1,
               reduce_mean: bool = True,
               descent: bool = True,
               norm_embeddings: bool = False):
    """Initializes DiscreteZOO with the information needed for our attack.

    Args:
      sampling_strategy: A function that accepts indices, embeddings, and
        a number to sample and returns replacement indices.
      num_to_sample: Integer to control how many samples are needed per estimate
      embeddings: embedding table as a tf.Tensor.
        <float32>[vocab_size, embedding_dim]
      adversarial_loss: A function that accepts the original sentence and our
        current sentence and returns a loss per sentence.
      reduce_mean: A boolean that flags how we want to reduce our displacements.
      descent: Boolean flag to determine if we're increasing or decreasing loss.
      norm_embeddings: Boolean flag to determine if embeddings should be normed
        for the discretization step.
    """
    self._sampling_strategy = sampling_strategy
    self._num_to_sample = num_to_sample
    self._embeddings = embeddings
    self._adversarial_loss = adversarial_loss
    self._reduce_mean = reduce_mean
    self._descent = descent
    self._norm_embeddings = norm_embeddings

    if self._norm_embeddings:
      self._embeddings = self._embeddings / tf.expand_dims(
          tf.norm(self._embeddings, axis=-1), -1)

  # pylint: disable=missing-function-docstring
  def _embedding_lookup(self, indices: tf.Tensor) -> tf.Tensor:
    return tf.nn.embedding_lookup(self._embeddings, indices)

  def _discretization(self, displacement: tf.Tensor,
                      current_tokens: tf.Tensor) -> tf.Tensor:
    r"""Converts a displacement vector to the ids of the best tokens.

    Discretizes the displacement vector into a score over tokens by using the
    method described in Hotflip.
    (Ebrahimi et al. 2018) https://arxiv.org/abs/1712.06751

    \frac{\partial L}{\partial indices} &=
      \frac{\partial L}{\partial embeddings}
      x \frac{\partial embeddings}{\partial indices}
      &= \frac{\partial L}{\partial embeddings} x embeddings

    \frac{\partial L}{\partial embeddings} is our averaged displacement vectors,
    so we just need to multiply the displacement vectors with our embedding
    matrix.

    Args:
      displacement: The expected scaled displacement over sampled replacements.
        <float32>[batch_size, embedding_dimension]
      current_tokens: The indices of the tokens we are currently at.
        <int32>[batch_size, 1]

    Returns:
      A <int32>[batch_size] tensor containing the index of the best replacement
        token for each sentence in the batch.
    """
    # Get dot-product similarity between displacement vectors and embeddings.
    # displacement is a matrix [batch_size, emb_dim] and embeddings is
    # [vocab_size, emb_dim]. embeddings needs to be transposed so that we have
    # [batch_size, emb_dim] @ [emb_dim, vocab_size] = [batch_size, vocab_size].
    displacement_token_similarities = tf.matmul(displacement,
                                                self._embeddings,
                                                transpose_b=True)
    # displacement_to_original_similarity is [batch_size, 1].
    displacement_to_original_similarity = tf.gather(
        displacement_token_similarities, current_tokens, batch_dims=1, axis=-1)
    # score_diff is [batch_size, vocab_size] and lowers displacement
    # similarities by the similarity of the current tokens.
    score_diff = displacement_token_similarities - \
      displacement_to_original_similarity
    # new_candidates are [batch_size,] and are vocab items most similar to
    # displacement vectors after subtracting the scores of the current tokens.
    new_candidates = tf.argmax(score_diff, axis=-1, output_type=tf.int32)
    # We need new candidates to be [batch_size, 1] for future scatter updates.
    return tf.expand_dims(new_candidates, -1)

  def _get_losses(self, sentences: tf.Tensor, indices: tf.Tensor,
                  sampled_tokens: tf.Tensor) -> tf.Tensor:
    """Helper function to loop over sampled tokens and compute loss for each.

    Args:
      sentences: A batch of sentences already numericalized.
        <int32>[batch_size, sentence_length]
      indices: A vector of indices, one for each sentence, that select which
        token in the sentences should be replaced.
        <int32>[batch_size, 1]
      sampled_tokens: A matrix of indices with num_to_sample for each sentence.
        <int32>[batch_size, num_to_sample]

    Returns:
      A matrix of losses for each sampled token in sampled_tokens.
        <float32>[batch_size, num_to_sample]
    """
    number_of_candidates = sampled_tokens.shape[-1]
    assert number_of_candidates == self._num_to_sample
    losses = []
    for candidate_id in range(number_of_candidates):
      token_swapped_sentence = self.scatter_helper(
          sentences, indices, sampled_tokens[:, candidate_id])
      # [batch_size, 1].
      per_item_losses = self._adversarial_loss(sentences,
                                               token_swapped_sentence)
      losses.append(per_item_losses)
    return tf.concat(losses, -1)

  @staticmethod
  def scatter_helper(sentences: tf.Tensor, indices: tf.Tensor,
                     new_values: tf.Tensor) -> tf.Tensor:
    """A helper function to make tensor_scatter_nd_update easier to use.

    This function creates a position tensor of size [batch_size, 2] to use with
    tensor_scatter_nd_update and keep all logic for scatter updates in one spot.

    Args:
      sentences: A batch of sentences already numericalized.
        <int32>[batch_size, sentence_length]
      indices: A vector of indices, one for each sentence, that select which
        token in the sentences should be replaced.
        <int32>[batch_size, 1]
      new_values: A vector of new values used to update sentences at location
        indices in each sentence. <int32>[batch_size, 1]

    Returns:
      A batch of sentences updated with new_values at indices in each sentence.
      <int32>[batch_size, sentence_length]
    """
    # tensor_scatter_nd_update requires coordinates for row and column.
    rows = tf.expand_dims(tf.range(sentences.shape[0], dtype=tf.int32), 1)
    # position contains the i,j coordinates for targets in sentence matrices.
    # [[0, 1, ..., batch_size], [indices[0], ..., indices[batch_size]]]
    position = tf.concat([rows, indices], -1)
    # Flatten our new values, because we are scattering scalars not slices.
    new_values = tf.reshape(new_values, (-1,))
    return tf.tensor_scatter_nd_update(sentences, position, new_values)

  def replace_token(self, sentences: tf.Tensor, indices: tf.Tensor,
                    iterations: int) -> tf.Tensor:
    """Replaces token at indexed position in sentences with loss reducing token.

    Args:
      sentences: A batch of sentences already numericalized.
        <int32>[batch_size, sentence_length]
      indices: A vector of indices, one for each sentence, that select which
        token in the sentences should be replaced.
        <int32>[batch_size, 1]
      iterations: How many iterations until we stop trying to find replacements.

    Returns:
      A <int32>[batch_size] tensor with a single token per sentence.
    """
    # Initializes the replacement_candidates to the sentence's original tokens.
    replacement_candidates = tf.gather(sentences,
                                       indices,
                                       batch_dims=1,
                                       axis=-1)

    for _ in range(iterations):
      # [batch_size, num_to_sample].
      sampled_tokens = self._sampling_strategy(replacement_candidates,
                                               self._embeddings,
                                               self._num_to_sample)
      # [batch_size, num_to_sample, emb_dim].
      sampled_embeddings = self._embedding_lookup(sampled_tokens)
      # [batch_size, 1, emb_dim].
      current_embeddings = self._embedding_lookup(replacement_candidates)
      # Get displacement vectors from current_embeddings to sampled_embeddings.
      displacement_vectors = sampled_embeddings - current_embeddings
      # Update the sentences with the current replacement_candidates.
      sentences_with_replacements = self.scatter_helper(sentences, indices,
                                                        replacement_candidates)
      # [batch_size, 1].
      current_loss = self._adversarial_loss(sentences_with_replacements,
                                            sentences)
      # [batch_size, num_to_sample].
      losses = self._get_losses(sentences, indices, sampled_tokens)
      # [batch_size, num_to_sample].
      loss_diff = losses - current_loss
      # norm is the l2 of the displacement_vectors and is
      # [batch_size, num_to_sample].
      norm = tf.norm(displacement_vectors, axis=-1)
      scaled_loss_diff = loss_diff / norm
      # Expand dims to make scaled_loss_diff [batch_size, num_to_sample, 1],
      # which is compatible with [batch_size, num_to_sample, emb_dim].
      scaled_loss_diff = tf.expand_dims(scaled_loss_diff, -1)
      # Yields a [batch_size, num_to_sample, emb_dim] of scaled displacements.
      gradient = scaled_loss_diff * displacement_vectors

      if self._descent:
        gradient = -gradient

      # Reduce over the first axis to make gradient [batch_size, emb_dim].
      if self._reduce_mean:
        reduced_gradient = tf.reduce_mean(gradient, axis=1)
      else:
        reduced_gradient = tf.reduce_sum(gradient, axis=1)

      replacement_candidates = self._discretization(reduced_gradient,
                                                    replacement_candidates)

    return replacement_candidates
