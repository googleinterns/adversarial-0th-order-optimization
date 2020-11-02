import csv
from collections import defaultdict
import itertools
from operator import itemgetter
from typing import Callable, Tuple, Dict, List

import more_itertools
import numpy as np
from nltk import tokenize
import pandas as pd
import tensorflow as tf
import transformers

from discretezoo.loss import adversarial_hinge


class ModelCallable:
  """ModelCallable wraps a neural network with its tokenizer and adds softmax.
  """

  def __init__(self,
               model_name: str,
               vocab: List[str],
               detokenizer: Callable[[List[str]], str],
               include_tokenizer: bool = False,
               padding_index: int = 0):
    """Initializes ModelCallable with the model and, optionally, its tokenizer.

    Arguments:
      model_name: A string containing either the path or name of a pretrained
        transformer model for the HuggingFace transformers library.
      vocab: A list of all tokens in the vocabulary.
      detokenizer: A callable that converts a list of tokens back into a string.
      include_tokenizer: A boolean that flags whether or not to include the
        tokenizer before calling the model.
      padding_index: An integer that is the index of the padding token in vocab.
    """
    self._model = (transformers.TFAutoModelForSequenceClassification.
                   from_pretrained(model_name))
    if include_tokenizer:
      self._model_tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_name, use_fast=True)
    else:
      self._model_tokenizer = None
    self._vocab = vocab
    self._detokenizer = detokenizer
    self.query_count = None
    self._padding_index = padding_index

  def reset_query_tracking(self, batch: tf.Tensor) -> None:
    """Re-initializes the query tracking vector with a zero vector.

    Arguments:
      batch: A batch of sentences. <int32>[batch_size, sentence_length]
    """
    self.query_count = tf.zeros((batch.shape[0],), dtype=tf.int32)

  def increment_query_count(self, batch: tf.Tensor) -> None:
    """For every sentence that isn't only padding tokens, increments the count.

    Arguments:
      batch: A batch of sentences, where some sentences may only be the padding
        token. This indicates that the sentence was removed inside attack loop.
        <int32>[batch_size, sentence_length]
    """
    # Converting the batch to numpy and checking for padding tokens on cpu was
    # faster than sending the tensors to the GPU to check for integer equality.
    batch = batch.numpy()
    is_not_padding_tokens = batch != self._padding_index
    is_not_empty_sentences = np.any(is_not_padding_tokens, axis=-1)
    self.query_count = self.query_count + tf.convert_to_tensor(
        is_not_empty_sentences, dtype=tf.int32)

  @property
  def get_query_count(self) -> tf.Tensor:
    return self.query_count

  @tf.function(experimental_relax_shapes=True)
  def _call_model_wrapper(self, **sentences):
    """Wraps the model call inside of a @tf.function call to improve speed.

    Arguments:
      **sentences: This is a dictionary of keyword arguments. Because models
        have different inputs and their tokenizers do as well, calling the model
        we need to be able to directly pass arbitrary inputs. Tensorflow's
        tf.function won't trace non-tensor arguments, so passing a dictionary
        directly would not work.
    """
    model_output = self._model(sentences)
    logits = model_output[0]
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities

  def __call__(self, sentences: tf.Tensor) -> tf.Tensor:
    """This optionally calls the tokenizer and passes sentences to the model.

    Arguments:
      sentences: Either a list of strings or a tensor containing a batch of
        sentences. If it is a tensor, then <int32>[batch_size, sentence_length].

    Returns:
      A tensor of probabilities for each item in the batch.
        <float32>[batch_size, number_of_classes]
    """

    if self.query_count is None:
      self.reset_query_tracking(sentences)
    self.increment_query_count(sentences)

    sentences = tensor_to_strings(sentences, self._vocab, self._detokenizer,
                                  self._padding_index)

    if self._model_tokenizer:
      # Assert that sentences is a non-empty list of strings.
      assert len(sentences) >= 1 and isinstance(sentences[0], str), \
        'Sentences must be a list of strings if using a tokenizer.'
      sentences = self._model_tokenizer(sentences,
                                        return_tensors='tf',
                                        padding=True,
                                        truncation=True)

    return self._call_model_wrapper(**sentences)


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
                                   output_type=labels.dtype)
    successful_attacks = adversarial_labels != labels
    return tf.expand_dims(successful_attacks, 1)


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
               original_probabilities: tf.Tensor) -> tf.Tensor:
    """Computes the difference between two sentences with KL Divergence.

    Arguments:
      changed_sentences: This is a copy of original_sentences with a single
        token per sentence dropped or masked.
        <int32>[batch_size, sentence_length]
      original_probabilities: This is the output probabilities for the original
        batch of sentences that changed sentences will be compared to.
        <float32>[batch_size, number_of_classes]

    Returns:
      A tensor containing the KL Divergence per sentence.
        <float32>[batch_size, 1]
    """
    changed_sentences_output = self._model_fun(changed_sentences)
    difference = tf.keras.losses.kl_divergence(original_probabilities,
                                               changed_sentences_output)
    return tf.reshape(difference, (-1, 1))


class AdversarialLoss:
  """AdversarialLoss is a callable class that combines distance and hinge loss.
  """

  def __init__(self,
               model_fun: Callable[[tf.Tensor], tf.Tensor],
               distance_fun: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
               embeddings: tf.Tensor,
               interpolation: float = 1.0,
               kappa: float = 0.0,
               tensorboard_logging: bool = False):
    """Initializes AdversarialLoss with parameters needed for loss calculation.

    Arguments:
      model_fun: A callable that returns probabilities for sentences.
      distance_fun: A callable implementing a distance metric between two
        sentences.
      embeddings: A tensor that contains the embeddings for measuring distance.
        <float32>[vocab_size, embedding_dimension]
      interpolation: A float that is multiplied with the distance before adding
        distance to the loss.
      kappa: A float that controls how certain a model needs to be about the
        adversarial class before achieving zero loss.
      tensorboard_logging: A boolean that controls if loss values are written
        to tensorboard.
    """
    self._distance_fun = distance_fun
    self._loss_fun = adversarial_hinge.untargeted_loss
    self._model_fun = model_fun
    self._interpolation = interpolation
    self._kappa = kappa
    self._tensorboard_logging = tensorboard_logging

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
    distance = self._distance_fun(original_sentences, adversarial_sentences)
    if self._tensorboard_logging:
      tf.summary.histogram('Hinge Loss',
                           loss,
                           step=tf.summary.experimental.get_step())
      tf.summary.histogram('Semantic Distance',
                           distance,
                           step=tf.summary.experimental.get_step())
    return loss + (self._interpolation * distance)


def load_embeddings(
    embeddings_path: str,
    vocab_path: str) -> Tuple[tf.Tensor, Dict[str, int], List[str]]:
  """This function loads embeddings into a tf.Tensor.

  We assume that the embeddings are saved using the method described in the
  Tensorflow tutorial:
  https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_learned_embeddings

  Arguments:
    embeddings_path: A string containing the path to a tsv file with an
      embedding per line.
    vocab_path: A string containing the path to a file with one token per line.

  Returns:
    embeddings: A tensor containing all of the embeddings.
      <float32>[vocab_size, embedding_dimension]
    token_to_id: A dictionary that maps each token to an index into embeddings.
    vocab: A list containing all tokens in the vocabulary.
  """
  with tf.io.gfile.GFile(embeddings_path) as embeddings_file:
    embeddings_csv = pd.read_csv(embeddings_file,
                                 quoting=csv.QUOTE_NONE,
                                 sep='\t',
                                 header=None)
  embeddings_matrix = tf.constant(embeddings_csv, dtype=tf.float32)
  with tf.io.gfile.GFile(vocab_path) as vocab_file:
    vocab = [line.strip() for line in vocab_file]
  indices = range(len(vocab))
  token_to_id = dict(zip(vocab, indices))
  return embeddings_matrix, token_to_id, vocab


def tokenize_sentence(text: str) -> List[str]:
  """This function accepts a single piece of text and tokenizes it.

  A single text can contain multiple sentences. However, the word tokenizer does
  not function over texts with multiple sentences. This splits texts into
  multiple sentences, runs the word tokenizer over each sentence, and then adds
  the sentences together to get a single list of tokens.

  Arguments:
    text: A string possibly containing multiple sentences.

  Returns:
    A list of tokens from the input text.
  """
  split_sentences = tokenize.sent_tokenize(text)
  tokenized_split_sentences = [
      tokenize.word_tokenize(sentence) for sentence in split_sentences
  ]
  joined_tokenized_sentences = list(itertools.chain(*tokenized_split_sentences))
  return joined_tokenized_sentences


def sort_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """This functions accepts a tf Dataset, and sorts it by the number of tokens.

  Arguments:
    dataset: A tf.data.Dataset where each example is a dictionary of tensors and
      each dictionary has the key 'sentence' whose value is a string tensor.

  Returns:
    A tf.data.Dataset sorted by the number of tokens contained in 'sentence',
      with the contents of 'sentence' being tokenized. Additionally, it adds the
      key 'token_count', which it uses to sort the dataset.
  """
  example_list = []

  for example in dataset:
    example['sentence'] = tokenize_sentence(
        example['sentence'].numpy().decode('utf-8'))
    example['token_count'] = len(example['sentence'])
    example_list.append(example)

  example_list = sorted(example_list, key=itemgetter('token_count'))

  example_dict = defaultdict(list)
  for example in example_list:
    for key in example.keys():
      example_dict[key].append(example[key])

  # Sentences have to be converted to ragged constants after they're all in the
  # same list, otherwise they become normal tensors whose shapes won't match
  # inside of Dataset.from_tensor_slices.
  example_dict['sentence'] = tf.ragged.constant(example_dict['sentence'])
  # Tensorflow Dataset initializer doesn't recognize defaultdicts.
  example_dict = dict(example_dict)
  return tf.data.Dataset.from_tensor_slices(example_dict)


def tensor_to_strings(numeric_tensor: tf.Tensor,
                      vocab: List[str],
                      detokenizer: Callable[[List[str]], str],
                      padding_index: int = 0) -> List[str]:
  """Converts a tensor of vocab indices into a list of strings.

  Arguments:
    numeric_tensor: A <int32>[batch_size, sentence_length] tensor of vocab
      indices.
    vocab: A list of tokens used to convert indices into tokens.
    detokenizer: A callable that takes a list of tokens and returns the tokens
      joined by whitespace and reverses the tokenization regexes.
    padding_index: The index of the padding token in the vocabulary, used to
      remove the padding tokens from the returned text.

  Returns:
    A list of detokenized sentences.
  """
  numeric_batch = numeric_tensor.numpy().tolist()
  padding_stripped_batch = [
      more_itertools.rstrip(token_ids,
                            lambda token_id: token_id == padding_index)
      for token_ids in numeric_batch
  ]

  token_batch = []
  for example in padding_stripped_batch:
    token_batch.append([vocab[index] for index in example])
  sentences = [detokenizer(sentence) for sentence in token_batch]
  return sentences
