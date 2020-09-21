"""Attack loop that optimizes multiple tokens in order of descending importance.
"""
import tensorflow as tf
from typing import Callable, Tuple

from discretezoo.attack import estimation


def loop(sentences: tf.Tensor, labels: tf.Tensor,
         optimizer: estimation.DiscreteZOO, token_importance_scores: tf.Tensor,
         early_stopping_criterion: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
         iterations_per_token: int,
         max_changes: int) -> Tuple[tf.Tensor, tf.Tensor]:
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
  sentence_length = sentences.shape[-1]
  # TODO: Consider allowing a token to be updated multiple times, exceeding
  # sentence_length number of changes.
  number_to_change = min(sentence_length, max_changes)
  adversarial_sentences = sentences
  # Sentinel value to keep track of which sentences don't need to be optimized.
  finished_attacks = tf.zeros((sentences.shape[0], 1), dtype=tf.bool)
  successful_adversarial_sentences = tf.zeros_like(sentences, dtype=tf.int32)

  for target_tokens in range(number_to_change):
    # attack_order[:, target_tokens] is [batch_size,], we need [batch_size, 1]
    indices = tf.expand_dims(attack_order[:, target_tokens], -1)
    replacement_tokens = optimizer.replace_token(sentences, original_sentences,
                                                 labels, indices,
                                                 iterations_per_token)
    adversarial_sentences = estimation.DiscreteZOO.scatter_helper(
        adversarial_sentences, indices, replacement_tokens)
    stoppable_attacks = early_stopping_criterion(adversarial_sentences, labels)
    if tf.reduce_any(stoppable_attacks):
      unfinished_attacks = tf.logical_not(finished_attacks)
      newly_finished_attacks = tf.logical_and(unfinished_attacks,
                                              stoppable_attacks)
      finished_attacks = tf.logical_or(newly_finished_attacks, finished_attacks)
      # Insert adversarial_sentences into successful_adversarial_sentences where
      # newly_finished_attacks is true.
      successful_adversarial_sentences = tf.where(
          newly_finished_attacks, adversarial_sentences,
          successful_adversarial_sentences)
      # Set adversarial_sentences to 0s where newly_finished_attacks is true.
      adversarial_sentences = tf.where(newly_finished_attacks, 0,
                                       adversarial_sentences)
    if tf.reduce_all(finished_attacks):
      # If they've all finished, we can return successful_adversarial_sentences.
      return successful_adversarial_sentences, finished_attacks

  # If all attacks haven't finished, we will return the unfinished attacks.
  unfinished_attacks = tf.logical_not(finished_attacks)
  adversarial_sentences = tf.where(unfinished_attacks, adversarial_sentences,
                                   successful_adversarial_sentences)
  return adversarial_sentences, finished_attacks
