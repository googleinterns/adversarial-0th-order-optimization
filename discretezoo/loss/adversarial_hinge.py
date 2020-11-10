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
"""Targeted and untargeted hinge loss functions for adversarial attacks."""
import tensorflow as tf


@tf.function
def targeted_loss(probabilities: tf.Tensor,
                  target_label: tf.Tensor,
                  kappa: float = 0.0) -> tf.Tensor:
  r"""Loss that is minimized when the target label is most likely.

  Implements equation 4 from the ZOO paper (Chen et al.)
  https://dl.acm.org/doi/10.1145/3128572.3140448
  f(x,t) = max(max_{i \neq t}(log[F(x)]_i) - log[F(x)]_t, −κ)

  The implementation is different than what is written in the paper. Here the
  loss is defined as
  f(x, t) = max(max_{i \neq t}(log[F(x)]_i) - log[F(x)]_t + k, 0), which moves
  the kappa to the other side of the maximum. This is that once an attack has
  been found that acheives kappa confidence, the original formula would return
  -kappa as a loss value. The new formula, on the other hand, returns a loss
  value of 0 once an attack with kappa confidence has been found.

  Args:
    probabilities: The softmax layer outputs of the target model.
      <float32>[batch_size, number_of_classes]
    target_labels: This is the label we are trying to make the model predict
      for each item in the batch. <int32>[batch_size, 1]
    kappa: Kappa controls how certain the target needs to be that the fake
      label is correct.

  Returns:
    A tensor <float32>[batch_size, 1] of per-example adversarial losses.
  """
  total_classes = probabilities.shape[1]
  batch_size = probabilities.shape[0]
  class_indices = tf.range(0, total_classes, dtype=tf.int32)
  log_probs = tf.math.log(probabilities)
  target_label = tf.reshape(target_label, (batch_size, 1))
  target_log_probs = log_probs[class_indices == target_label]
  # Using a boolean mask flattens the tensor, this restores the needed shape.
  target_log_probs = tf.reshape(target_log_probs, (batch_size, 1))
  # There is no function that gathers everything but the index so this uses
  # indices != target_label to select everything but target_label.
  other_log_probs = log_probs[class_indices != target_label]
  other_log_probs = tf.reshape(other_log_probs, (batch_size, total_classes - 1))
  max_other_log_probs = tf.reduce_max(other_log_probs, axis=-1, keepdims=True)
  return tf.math.maximum(max_other_log_probs - target_log_probs + kappa, 0)


@tf.function
def untargeted_loss(probabilities: tf.Tensor, true_label: tf.Tensor,
                    kappa: float) -> tf.Tensor:
  r"""Loss that is minimized when any label is more likely than the true label.

  Implements equation 5 from the ZOO paper (Chen et al.)
  https://dl.acm.org/doi/10.1145/3128572.3140448
  f(x,t) = max(log[F(x)]_t − max_{i \neq t}(log[F(x)]_i), −κ)

  The implementation is different than what is written in the paper. Here the
  loss is defined as
  f(x,t) = max(log[F(x)]_t − max_{i \neq t}(log[F(x)]_i) + k, 0)
  the kappa to the other side of the maximum. This is that once an attack has
  been found that acheives kappa confidence, the original formula would return
  -kappa as a loss value. The new formula, on the other hand, returns a loss
  value of 0 once an attack with kappa confidence has been found.

  Args:
    probabilities: The softmax layer outputs of the target model.
      <float32>[batch_size, number_of_classes]
    true_label: This is the true label for each item in the batch.
      <int32>[batch_size, 1]
    kappa: Kappa controls how certain the target needs to be that the fake label
      is correct.

  Returns:
    A tensor <float32>[batch_size, 1] of per-example adversarial losses.
  """
  total_classes = probabilities.shape[1]
  batch_size = probabilities.shape[0]
  class_indices = tf.range(0, total_classes, dtype=tf.int32)
  log_probs = tf.math.log(probabilities)
  true_label = tf.reshape(true_label, (batch_size, 1))
  true_log_probs = log_probs[class_indices == true_label]
  # Using a boolean mask flattens the tensor, this restores the needed shape.
  true_log_probs = tf.reshape(true_log_probs, (batch_size, 1))
  # There is no function that gathers everything but the index so this uses
  # indices != true_label to select everything but true_label.
  other_log_probs = log_probs[class_indices != true_label]
  other_log_probs = tf.reshape(other_log_probs, (batch_size, total_classes - 1))
  max_other_log_probs = tf.reduce_max(other_log_probs, axis=-1, keepdims=True)
  return tf.math.maximum(true_log_probs - max_other_log_probs + kappa, 0)
