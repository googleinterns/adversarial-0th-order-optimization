"""Contains the importance scorer for deciding which token to change."""
import tensorflow as tf
from typing import Callable, Dict, Any


def mask_position(sentences: tf.Tensor, position: int, **kwargs) -> tf.Tensor:
  """Masks a single position in all sentences with a given integer id.

  Arguments:
    sentences: This is a batch of sentences <int32>[batch_size, length].
    position: Position is the index of the token in sentences to be masked.
    kwargs: A dict that must contain a value for the key 'mask_id'.

  Returns:
    A batch of sentences with the token at position replaced with mask_id.
  """
  assert 'mask_id' in kwargs
  assert position >= 0
  assert position < sentences.shape[1]
  mask_vector = tf.reshape(
      tf.constant([[kwargs['mask_id']] * sentences.shape[0]]), (-1, 1))
  masked_sentences = tf.concat(
      [sentences[:, :position], mask_vector, sentences[:, position + 1:]], 1)
  return masked_sentences


def drop_position(sentences: tf.Tensor, position: int, **kwargs) -> tf.Tensor:
  """Drops a single token at position from every sentence in sentences.

  Arguments:
    sentences: This is a batch of sentences <int32>[batch_size, length].
    position: Position is the index of the token in sentences to be dropped.

  Returns:
    A batch of sentences with the token at position removed.
  """
  assert position >= 0
  assert position < sentences.shape[1]
  dropped_sentences = tf.concat(
      [sentences[:, :position], sentences[:, position + 1:]], 1)
  return dropped_sentences


def scorer(sentences: tf.Tensor,
           original_probabilities: tf.Tensor,
           output_difference_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
           removal_fn: Callable[[tf.Tensor, int], tf.Tensor],
           removal_fn_kwargs: Dict[str, Any] = {},
           pad_id: int = 0) -> tf.Tensor:
  """Performs importance scoring for each token in the input sentences.

  This function loops over every token in the input sentences and either drops
  it or masks it.

  Args:
    sentences: A tensor <int32>[batch_size, sentence_length] of token ids.
      Sentences of different lengths are assumed to be padded with pad_id.
    original_probabilities: The class probabilities for the original sentences.
      Pre-computing the probabilities outside of output_difference_fn lets them
      be re-used, which reduces model queries and saves on compute time.
      <float32>[batch_size, number_of_classes]
    output_difference_fn: A callable that returns the output difference between
      inputs. First input is the sentences with the dropped/masked token.
      Second input is the original sentences. Outputs are in [0, +inf]. Must
      return a <float32>[batch_size, 1] tensor.
    removal_fn: A callable that modifies a single position in the input
      sentences so that the importance of that position can be evaluated.
    removal_fn_kwargs: A dictionary of keyword arguments for removal_fn.
    pad_id: The index of the padding token in the vocabulary. This is used so
      padded positions can't receive non-zero importance scores.

  Returns:
    A matrix containing the token importance scores for each sentence.
      <float32>[batch_size, sentence_length]
  """
  position_mask = tf.where(sentences == pad_id, 0.0, 1.0)
  differences = []
  for position in range(sentences.shape[-1]):
    dropped_token_sentences = removal_fn(sentences, position,
                                         **removal_fn_kwargs)
    loss_differences = output_difference_fn(dropped_token_sentences,
                                            original_probabilities)
    differences.append(loss_differences)

  difference_matrix = tf.concat(differences, -1)
  return position_mask * difference_matrix
