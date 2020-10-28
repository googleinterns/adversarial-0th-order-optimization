import tensorflow as tf


class EmbeddedCosineDistance:
  """EmbeddedCosineDistance calculates cosine distance in embedding space.

  Attributes:
    embeddings: A tensor containing an embedding vector for each index in vocab.
      <float32>[vocab_size, embedding_dimension]
  """

  def __init__(self, embeddings: tf.Tensor):
    """Initializes EmbeddedCosineDistance with embeddings.

    Arguments:
      embeddings: A tensor containing an embedding for each index in vocab.
        <float32>[vocab_size, embedding_dimension]
    """
    assert embeddings.ndim == 2, (
        'Embeddings are expected to have 2 dimensions'
        f' but you passed a tensor with {embeddings.ndim}.')
    self._embeddings = embeddings

  @tf.function
  def __call__(self, original_sentences: tf.Tensor,
               adversarial_sentences: tf.Tensor) -> tf.Tensor:
    r"""Calculates cosine distance between reduced embedded sentences.

    Sentences are embedded and then reduced by summing them up.
    Cosine similarity is then given by \frac{v_{original} \cdot v_{adversarial}}
      {|v_{original}| \times |v_{adversarial|}}.
    Cosine distance is defined as 1 - similarity.

    Arguments:
      original_sentences: A tensor of token indices in the original sentences.
        <int32>[batch_size, sentence_length]
      adversarial_sentences: A tensor of token indices in the adversarial
        sentences. <int32>[batch_size, sentence_length]

    Returns:
      A tensor <float32>[batch_size, 1] of cosine distances between original and
        adversarial sentences. Return values are in the range [0, 2]
        https://www.tensorflow.org/api_docs/python/tf/keras/losses/cosine_similarity
        is used internally, which computes negative similarity, and 1 is added.
    """
    original_sentences_embedded = tf.nn.embedding_lookup(
        self._embeddings, original_sentences)
    adversarial_sentences_embedded = tf.nn.embedding_lookup(
        self._embeddings, adversarial_sentences)

    original_sentences_reduced = tf.math.reduce_sum(original_sentences_embedded,
                                                    axis=1)
    adversarial_sentences_reduced = tf.math.reduce_sum(
        adversarial_sentences_embedded, axis=1)

    # Unintuitively, tf.keras.losses.cosine_similarity returns distances in
    # [-1, 1]. Adding 1 means that two vectors will have 0 as a minimum distance
    # instead of -1, which is helpful in later loss computation.
    distance = 1 + tf.keras.losses.cosine_similarity(
        original_sentences_reduced, adversarial_sentences_reduced)
    return tf.expand_dims(distance, 1)


class EmbeddedEuclideanDistance:
  """EmbeddedEuclideanDistance calculates euclidean distance in embedding space.

  Attributes:
    embeddings: A tensor containing an embedding vector for each index in vocab.
      <float32>[vocab_size, embedding_dimension]
    reduce_mean: This is a boolean flag that signals how embedded sentences will
      be reduced to a single vector. True for mean, False for sum.
  """

  def __init__(self, embeddings: tf.Tensor, reduce_mean: bool = True):
    """Initializes EmbeddedEuclideanDistance with embeddings and reduction type.

    Arguments:
      embeddings: A tensor containing an embedding for each index in vocab.
        <float32>[vocab_size, embedding_dimension]
      reduce_mean: This boolean flag signals how embedded sentences will be
        reduced to a single vector. True for mean, False for sum.
    """
    assert embeddings.ndim == 2, (
        'Embeddings are expected to have 2 dimensions'
        f' but you passed a tensor with {embeddings.ndim}.')
    self._embeddings = embeddings
    self._reduce_mean = reduce_mean

  @tf.function
  def __call__(self, original_sentences: tf.Tensor,
               adversarial_sentences: tf.Tensor) -> tf.Tensor:
    """Calculates euclidean distances between reduced embedded sentences.

    Arguments:
      original_sentences: A tensor of token indices in the original sentences.
        <int32>[batch_size, sentence_length]
      adversarial_sentences: A tensor of token indices in the adversarial
        sentences. <int32>[batch_size, sentence_length]

    Returns:
      A tensor <float32>[batch_size, 1] of euclidean distances between original
        and adversarial sentences.
    """
    original_sentences_embedded = tf.nn.embedding_lookup(
        self._embeddings, original_sentences)
    adversarial_sentences_embedded = tf.nn.embedding_lookup(
        self._embeddings, adversarial_sentences)

    if self._reduce_mean:
      original_sentences_reduced = tf.math.reduce_mean(
          original_sentences_embedded, axis=1)
      adversarial_sentences_reduced = tf.math.reduce_mean(
          adversarial_sentences_embedded, axis=1)
    else:
      original_sentences_reduced = tf.math.reduce_sum(
          original_sentences_embedded, axis=1)
      adversarial_sentences_reduced = tf.math.reduce_sum(
          adversarial_sentences_embedded, axis=1)

    difference_vector = tf.math.subtract(original_sentences_reduced,
                                         adversarial_sentences_reduced)
    distance = tf.norm(difference_vector, axis=-1, keepdims=True)
    return distance
