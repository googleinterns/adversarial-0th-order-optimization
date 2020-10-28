from sklearn.metrics import pairwise
import tensorflow as tf


def uniform_sampling(indices: tf.Tensor, embeddings: tf.Tensor,
                     num_to_sample: int):
  """Samples from the vocabulary uniformly without being informed by embeddings.

  Args:
    indices: The indices of the current tokens. Because the sampling is uniform
      over the vocabulary, no information from the current indices are used.
      However, they are still needed in order to have the correct sized output.
      <int32>[batch_size, 1]
    embeddings: Like indices, the embeddings' information are not used in this
      sampling function, but they are used to specify the range to sample in.
      <float32>[vocab_size, embedding_dimension]
    num_to_sample: The number of alternative tokens we would like to sample.

  Returns:
    A tensor <int32>[batch_size, num_to_sample] of sampled token ids.
  """
  vocab_size = embeddings.shape[0]
  batch_size = indices.shape[0]
  return tf.random.uniform((batch_size, num_to_sample),
                           maxval=vocab_size,
                           dtype=tf.int32)


def knn_sampling(indices: tf.Tensor, embeddings: tf.Tensor, num_to_sample: int,
                 distance_metric: str) -> tf.Tensor:
  """Samples from the vocabulary by taking ids of the most similar embeddings.

  Similarity is given by the euclidean or cosine distance between embeddings.
  This function uses sklearn pairwise to find the most similar embedding over
  the entire vocabulary to our current embedding.

  Args:
    indices: The indices of the current tokens. <int32>[batch_size, 1]
    embeddings: The embeddings used to guide the sampling of tokens.
      <float32>[vocab_size, embedding_dimension]
    num_to_sample: The number of alternative tokens we would like to sample.
    distance_metric: Which distance metric to use, (e.g. "cosine" or
      "euclidean"). For a complete list of metrics, please refer to the
      documentation for sklearn.
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn-metrics-pairwise-distances

  Returns:
    A tensor <int32>[batch_size, num_to_sample] of sampled token ids.
  """
  # Flattens indices from [batch_size, 1] to [batch_size,], so that after
  # embedding, we will have [batch_size, embedding_dim], which sklearn needs.
  indices = tf.reshape(indices, (-1,))
  embedded_indices = tf.nn.embedding_lookup(embeddings, indices)
  distances = pairwise.pairwise_distances(embedded_indices,
                                          Y=embeddings,
                                          metric=distance_metric)
  # Taking top_k negative distances yields closest tokens, instead of furthest.
  # We are taking k + 1 because embeddings are closest to themselves, so to have
  # k, we will need to remove the top 1.
  _, top_k_indices = tf.math.top_k(-distances, k=num_to_sample + 1)
  # This slices the top 1 off the second dimension yielding [batch_size, k].
  return top_k_indices[:, 1:]


def knn_sampling_cosine(indices: tf.Tensor, embeddings: tf.Tensor,
                        num_to_sample: int):
  """Samples from the vocabulary by taking ids of the most similar embeddings.

  Similarity is given by the cosine similarity between two embeddings. This
  function uses sklearn pairwise to find the most similar embedding over the
  entire vocabulary to our current embedding.

  Args:
    indices: The indices of the current tokens. <int32>[batch_size, 1]
    embeddings: The embeddings used to guide the sampling of tokens.
      <float32>[vocab_size, embedding_dimension]
    num_to_sample: The number of alternative tokens we would like to sample.

  Returns:
    A tensor <int32>[batch_size, num_to_sample] of sampled token ids.
  """
  return knn_sampling(indices, embeddings, num_to_sample, 'cosine')


def knn_sampling_euclidean(indices: tf.Tensor, embeddings: tf.Tensor,
                           num_to_sample: int):
  """Samples from the vocabulary by taking ids of the most similar embeddings.

  Similarity is given by the euclidean distance between two embeddings. This
  function uses sklearn pairwise to find the most similar embedding over the
  entire vocabulary to our current embedding.

  Args:
    indices: The indices of the current tokens. <int32>[batch_size, 1]
    embeddings: The embeddings used to guide the sampling of tokens.
      <float32>[vocab_size, embedding_dimension]
    num_to_sample: The number of alternative tokens we would like to sample.

  Returns:
    A tensor <int32>[batch_size, num_to_sample] of sampled token ids.
  """
  return knn_sampling(indices, embeddings, num_to_sample, 'euclidean')
