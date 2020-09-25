"""Attack loop that optimizes multiple tokens in order of descending importance.
"""
from typing import Callable, Tuple

import tensorflow as tf

from discretezoo.attack import estimation


def loop(sentences: tf.Tensor,
         labels: tf.Tensor,
         optimizer: estimation.DiscreteZOO,
         token_importance_scores: tf.Tensor,
         early_stopping_criterion: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
         iterations_per_token: int,
         max_changes: int,
         padding_index: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
  """Implementation of main attack loop which adversarially optimizes sentences.

  Args:
    sentences: A tensor containing indices of tokens in the vocab to optimize.
      <int32>[batch_size, sentence_length]
    labels: A tensor containing per-example labels for each sentence. The labels
      are the model's output labels on the original sentences.
      <int32>[batch_size, 1]
    optimizer: An instance of DiscreteZOO that will replace tokens in sentences.
    token_importance_scores: A tensor of token importance scores in sentences.
      <int32>[batch_size, sentence_length]
    early_stopping_criterion: A function that takes the adversarial sentences
      and the original sentences and returns a boolean per sentence indicating
      whether or not the sentence has flipped the output label.
    iterations_per_token: A limit for the loop inside the optimizer, how many
      times a token can be updated.
    max_changes: An upper limit on how many tokens per sentence can be changed.
      If max_changes is 0, all tokens in the sentence are able to be changed.
    padding_index: The index in vocab which corresponds to the padding token.
      This is used so that padding tokens are not changed in the attack.

  Returns:
    adversarial_sentences: An <int32>[batch_size, sentence_length] tensor of
      adversarial sentences. Contains unfinished attacks if unsuccessful.
    finished_attacks: A <bool>[batch_size, 1] tensor indicating if the
      corresponding adversarial sentence was successful or not.
  """
  attack_order = tf.argsort(token_importance_scores,
                            axis=-1,
                            direction='DESCENDING')
  # TODO: Re-evaluate importance after token change instead of it being static.
  original_sentences = tf.identity(sentences)
  batch_size = sentences.shape[0]
  not_padding_tokens = tf.cast(sentences != padding_index, tf.int32)
  sentence_lengths = tf.reduce_sum(not_padding_tokens, axis=-1, keepdims=True)
  # TODO: Consider allowing a token to be updated multiple times, exceeding
  # sentence_length number of changes.
  if max_changes == 0:
    per_sentence_tokens_to_change = sentence_lengths
  else:
    per_sentence_tokens_to_change = tf.minimum(sentence_lengths, max_changes)

  max_tokens_to_change = tf.reduce_max(per_sentence_tokens_to_change).numpy()

  adversarial_sentences = sentences
  # Sentinel values to keep track of which sentences don't need to be optimized.
  successful_attacks = tf.zeros((batch_size, 1), dtype=tf.bool)
  stopped_attacks = tf.zeros((batch_size, 1), dtype=tf.bool)
  stopped_attacks_storage = tf.zeros_like(sentences)

  for target_tokens in range(max_tokens_to_change):
    # attack_order[:, target_tokens] is [batch_size,], we need [batch_size, 1]
    indices = tf.expand_dims(attack_order[:, target_tokens], -1)
    replacement_tokens = optimizer.replace_token(adversarial_sentences,
                                                 original_sentences, labels,
                                                 indices, iterations_per_token)
    adversarial_sentences = estimation.DiscreteZOO.scatter_helper(
        adversarial_sentences, indices, replacement_tokens)
    finished_attacks = early_stopping_criterion(adversarial_sentences, labels)
    # An attack is only successful if it hasn't already reached the max changes.
    # This is to prevent a zeroed out sentence from being seen as successful.
    # If target_tokens is now equal to a value in per_sentence_tokens_to_change
    # we need to remove the corresponding sentence from the batch.
    max_changes_reached = per_sentence_tokens_to_change == (target_tokens + 1)

    stoppable_attacks = tf.logical_or(max_changes_reached, finished_attacks)
    if tf.reduce_any(stoppable_attacks):
      running_attacks = tf.logical_not(stopped_attacks)
      newly_stopped_attacks = tf.logical_and(running_attacks, stoppable_attacks)
      stopped_attacks = tf.logical_or(newly_stopped_attacks, stopped_attacks)
      successful_attacks = tf.logical_or(successful_attacks, finished_attacks)
      # Insert adversarial_sentences into stopped_attacks where
      # newly_finished_attacks is true.
      stopped_attacks_storage = tf.where(newly_stopped_attacks,
                                         adversarial_sentences,
                                         stopped_attacks_storage)
      # Set adversarial_sentences to padding if stopped_attacks is true.
      adversarial_sentences = tf.where(stopped_attacks, padding_index,
                                       adversarial_sentences)
    if tf.reduce_all(successful_attacks):
      # If they've all finished, we can return stopped_attacks.
      return stopped_attacks_storage, successful_attacks

  # If all attacks haven't finished, we will return the unfinished attacks.
  running_attacks = tf.logical_not(stopped_attacks)
  adversarial_sentences = tf.where(running_attacks, adversarial_sentences,
                                   stopped_attacks_storage)

  return adversarial_sentences, successful_attacks
