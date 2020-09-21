import tensorflow as tf
import transformers
import pandas as pd
import csv
from discretezoo.loss import semantic_similarity, adversarial_hinge
from typing import Union, Callable, Tuple, Dict, List


class ModelCallable:
  """ModelCallable wraps a neural network with its tokenizer and adds softmax.
  """

  def __init__(self,
               model_name: str,
               vocab: List[str],
               detokenizer: Callable[[List[str]], str],
               include_tokenizer: bool = False):
    """Initializes ModelCallable with the model and, optionally, its tokenizer.

    Arguments:
      model_name: A string containing either the path or name of a pretrained
        transformer model for the HuggingFace transformers library.
      vocab: A list of all tokens in the vocabulary.
      detokenizer: A callable that converts a list of tokens back into a string.
      include_tokenizer: A boolean that flags whether or not to include the
        tokenizer before calling the model.
    """
    self._model = transformers.TFBertForSequenceClassification.from_pretrained(
        model_name)
    if include_tokenizer:
      self._model_tokenizer = transformers.BertTokenizer.from_pretrained(
          model_name)
    else:
      self._model_tokenizer = None

    self._vocab = vocab
    self._detokenizer = detokenizer

  def __call__(self, sentences: Union[List[str], tf.Tensor]) -> tf.Tensor:
    """This optionally calls the tokenizer and passes sentences to the model.

    Arguments:
      sentences: Either a list of strings or a tensor containing a batch of
        sentences. If it is a tensor, then <int32>[batch_size, sentence_length].

    Returns:
      A tensor of probabilities for each item in the batch.
        <float32>[batch_size, number_of_classes]
    """
    if isinstance(sentences, tf.Tensor):
      # pytype complains that List doesn't have numpy as a method. Checking that
      # it's a tensor first means that is not a problem.
      # pytype: disable=attribute-error
      numeric_batch = sentences.numpy().tolist()
      # pytype: enable=attribute-error
      token_batch = [[self._vocab[index]
                      for index in sentence]
                     for sentence in numeric_batch]
      sentences = [self._detokenizer(sentence) for sentence in token_batch]

    if self._model_tokenizer:
      # Assert that sentences is a non-empty list of strings.
      assert len(sentences) >= 1 and isinstance(sentences[0], str), \
        "Sentences must be a list of strings if using a tokenizer."
      sentences = self._model_tokenizer(sentences, return_tensors='tf')
    model_output = self._model(sentences, return_dict=True)
    logits = model_output['logits']
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities


class EarlyStopping:
  """EarlyStopping is the criterion that determines if an attack can be stopped.
  """

  def __init__(self, model_fun: Callable[[tf.Tensor], tf.Tensor]):
    """Initializes EarlyStopping with the model function and attack type.

    Arguments:
      model_fun: The function that accepts a sentence and returns a distribution
        over class labels.
    """
    self._model_fun = model_fun

  def __call__(self, adversarial_sentences: tf.Tensor,
               labels: tf.Tensor) -> tf.Tensor:
    """This implements the criterion that determines if the attack is finished.

    Arguments:
      adversarial_sentences: A batch of sentences that is being optimized.
        <int32>[batch_size, sentence_length]
      labels: A label for each sentence in the batch of adversarial_sentences.
        <int32>[batch_size, 1]

    Returns:
      A batch of booleans that determines if each example is finished yet.
        <bool>[batch_size, 1]
    """
    adversarial_probabilities = self._model_fun(adversarial_sentences)
    adversarial_labels = tf.argmax(adversarial_probabilities,
                                   axis=-1,
                                   output_dtype=labels.dtype)
    adversarial_labels = tf.reshape(adversarial_labels, (-1, 1))
    return adversarial_labels != labels


class OutputDifference:
  """This class calculates differences in model output at two different points.

  The difference between two points is defined as the KL Divergence between the
  model's output probability distribution at those points. KL Divergence is not
  symmetric, so the input order matters. In this case the probability
  distribution for the original sentences is given as the true distribution.
  """

  def __init__(self, model_fun: Callable[[tf.Tensor], tf.Tensor]):
    """Initializes OutputDifference with the model function.

    Arguments:
      model_fun: A callable that accepts sentences and returns probabilities.
    """
    self._model_fun = model_fun

  def __call__(self, changed_sentences: tf.Tensor,
               original_sentences: tf.Tensor) -> tf.Tensor:
    """Computes the difference between two sentences with KL Divergence.

    Arguments:
      changed_sentences: This is a copy of original_sentences with a single
        token per sentence dropped or masked.
        <int32>[batch_size, sentence_length]
      original_sentences: This is the original batch of sentences that changed
        sentences will be compared to.
        <int32>[batch_size, sentence_length]

    Returns:
      A tensor containing the KL Divergence per sentence.
        <float32>[batch_size, 1]
    """
    changed_sentences_output = self._model_fun(changed_sentences)
    original_sentences_output = self._model_fun(original_sentences)
    difference = tf.keras.losses.kl_divergence(original_sentences_output,
                                               changed_sentences_output)
    return tf.reshape(difference, (-1, 1))


class AdversarialLoss:
  """AdversarialLoss is a callable class that combines distance and hinge loss.
  """

  def __init__(self,
               model_fun: Callable[[tf.Tensor], tf.Tensor],
               use_cosine: bool,
               embeddings: tf.Tensor,
               interpolation: float = 1.0,
               kappa: float = 0.0):
    """Initializes AdversarialLoss with parameters needed for loss calculation.

    Arguments:
      model_fun: A callable that returns probabilities for sentences.
      use_cosine: A boolean that controls which distance function is used to
        measure semantic similarity for the attack. True means use cosine
        distance and false means use euclidean distance.
      embeddings: A tensor that contains the embeddings for measuring distance.
        <float32>[vocab_size, embedding_dimension]
      interpolation: A float that is multiplied with the distance before adding
        distance to the loss.
      kappa: A float that controls how certain a model needs to be about the
        adversarial class before achieving zero loss.
    """
    if use_cosine:
      distance_object = semantic_similarity.EmbeddedCosineDistance(embeddings)
    else:
      distance_object = semantic_similarity.EmbeddedEuclideanDistance(
          embeddings)

    self._distance_object = distance_object
    self._loss_fun = adversarial_hinge.untargeted_loss
    self._model_fun = model_fun
    self._interpolation = interpolation
    self._kappa = kappa

  def __call__(self, original_sentences: tf.Tensor, labels: tf.Tensor,
               adversarial_sentences: tf.Tensor) -> tf.Tensor:
    """This function computes the adversarial loss.

    Implements equation 1 from the ZOO paper (Chen et al.)
    https://dl.acm.org/doi/10.1145/3128572.3140448
    Adversarial loss is defined as hinge loss + (interpolation * distance).

    Arguments:
      original_sentences: A batch of sentences that adversarial_sentences will
        be compared with to calculate distances.
        <int32>[batch_size, sentence_length]
      labels: This are the labels per item in the batch for original sentences.
        <int32>[batch_size, 1]
      adversarial_sentences: A batch of sentences that we are optimizing.
        <int32>[batch_size, sentence_length]

    Returns:
      The adversarial loss as defined above. <float32>[batch_size, 1]
    """
    adversarial_probabilities = self._model_fun(adversarial_sentences)
    loss = self._loss_fun(adversarial_probabilities, labels, self._kappa)
    distance = self._distance_object(original_sentences, adversarial_sentences)
    return loss + (self._interpolation * distance)


def load_embeddings(
    embeddings_path: str,
    vocab_path: str) -> Tuple[tf.Tensor, Dict[str, int], List[str]]:
  """This function loads embeddings into a tf.Tensor.

  We assume that the embeddings are saved using the method described in the
  Tensorflow tutorial:
  https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_learned_embeddings

  Arguments:
    path: A string containing the path of a tsv file with an embedding per line.
    vocab: A string containing the path of a text file with a token per line.

  Returns:
    embeddings: A tensor containing all of the embeddings.
      <float32>[vocab_size, embedding_dimension]
    token_to_id: A dictionary that maps each token to an index into embeddings.
    vocab: A list containing all tokens in the vocabulary.
  """
  with tf.io.gfile.GFile(embeddings_path) as embeddings_file:
    embeddings_csv = pd.read_csv(embeddings_file,
                                 quoting=csv.QUOTE_NONE,
                                 sep='\t')
  embeddings_matrix = tf.constant(embeddings_csv, dtype=tf.float32)
  with tf.io.gfile.GFile(vocab_path) as vocab_file:
    vocab = [line.strip() for line in vocab_file]

  indices = range(len(vocab))
  token_to_id = dict(zip(vocab, indices))
  return embeddings_matrix, token_to_id, vocab
