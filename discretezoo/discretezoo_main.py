import datetime
import os
import itertools

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from nltk import tokenize
from nltk.tokenize import treebank

from discretezoo import attack_setup
from discretezoo.attack import importance, attack_loop, estimation, sampling

from typing import List

FLAGS = flags.FLAGS
# Target model settings.
flags.DEFINE_string('model', None, 'The directory of the model to attack.')
flags.DEFINE_integer('padding_index',
                     0,
                     'Which index to use as the padding.',
                     lower_bound=0)
flags.DEFINE_integer('oov_index',
                     0,
                     'Which index to use for unknown tokens.',
                     lower_bound=0)
flags.DEFINE_boolean(
    'include_tokenizer', True,
    'Include the pretrained model\'s tokenizer in the call function.')
# Optimizer settings.
flags.DEFINE_integer(
    'token_changes_per_sentence',
    0, 'This controls how many tokens per sentence can be changed.\n'
    'If this is set to 0, all tokens in a sentence may be changed.',
    lower_bound=0)
flags.DEFINE_integer(
    'changes_per_token',
    3,
    'This controls how many times a token can be changed by the optimizer.',
    lower_bound=1)
flags.DEFINE_string(
    'embeddings_file', None, 'The path to a tsv file containing embeddings.\n'
    'Vectors have a corresponding token in vocab_file on the same line number.')
flags.DEFINE_string(
    'vocab_file', None,
    'The path to a text file containing an individual vocab item on each line.')
flags.DEFINE_enum(
    'sampling_strategy', 'uniform', ['uniform', 'knn_euclidean', 'knn_cosine'],
    'Which sampling method to use to replace tokens in sentences.')
flags.DEFINE_integer('num_to_sample',
                     1,
                     'How many tokens to sample while estimating the gradient.',
                     lower_bound=0)
flags.DEFINE_bool('normalize_embeddings', False,
                  'Normalize embeddings used by the optimizer.')
flags.DEFINE_bool(
    'reduce_mean', True,
    'Controls whether sentences and gradients are reduced using mean or sum.')
# Attack settings.
flags.DEFINE_string(
    'dataset', None,
    'The name of the dataset you would like to make adversarial.\n'
    'It must be the name of a valid dataset in tensorflow_datasets.')
flags.DEFINE_string('split', 'test', 'Which split of the dataset to use.')
flags.DEFINE_integer(
    'num_examples',
    0, 'The number of sentences in the dataset to make adversarial. \n'
    '0 means all sentences. The attack will start with the first sentence in '
    'the dataset and attack this many sentences.',
    lower_bound=0)
flags.DEFINE_integer('batch_size',
                     8,
                     'How many sentences to attack simultaneously.',
                     lower_bound=1)
flags.DEFINE_enum(
    'semantic_similarity', 'cosine', ['euclidean', 'cosine'],
    'This controls how similarity between two sentences is computed.')
flags.DEFINE_float(
    'interpolation',
    1.0,
    'Interpolation factor between adversarial loss and semantic similarity.',
    lower_bound=0.0)
flags.DEFINE_float(
    'kappa',
    0.0,
    'Controls how confident the model should be about the adversarial label.',
    lower_bound=0.0)

# Logging Settings
flags.DEFINE_string('output_file', None,
                    'The output file to write adversarial examples to.')
flags.DEFINE_string('tensorboard_dir', None,
                    'The output directory to write tensorboard logs to.')

flags.mark_flags_as_required(
    ['model', 'embeddings_file', 'vocab_file', 'dataset', 'output_file'])


def tokenize_sentences(text_list: List[str]) -> List[List[str]]:
  """This function accepts a list of texts and tokenizes them.

  A single text can contain multiple sentences, however the word tokenizer does
  not function over texts with multiple sentences. This splits texts into
  multiple sentences, runs the word tokenizer over each sentence, and then adds
  the sentences together to get a single list of tokens.

  Arguments:
    test_list: A list of strings where each string is a text possibly containing
      multiple sentences.

  Returns:
    A list of list of strings where the strings are tokens from the input texts.
  """
  split_sentences = [tokenize.sent_tokenize(text) for text in text_list]
  tokenized_split_sentences = []
  for sentences in split_sentences:
    tokenized_sentences = [
        tokenize.word_tokenize(sentence) for sentence in sentences
    ]
    tokenized_split_sentences.append(tokenized_sentences)
  joined_tokenized_sentences = [
      list(itertools.chain(*sentences))
      for sentences in tokenized_split_sentences
  ]
  return joined_tokenized_sentences


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tokenizer = treebank.TreebankWordTokenizer()
  detokenizer = treebank.TreebankWordDetokenizer()

  if FLAGS.tensorboard_dir:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.tensorboard_dir, current_time))

  embeddings, token_to_id, vocab = attack_setup.load_embeddings(
      FLAGS.embeddings_file, FLAGS.vocab_file)

  model_fun = attack_setup.ModelCallable(
      FLAGS.model,
      vocab,
      detokenizer.detokenize,
      include_tokenizer=FLAGS.include_tokenizer)
  adversarial_loss = attack_setup.AdversarialLoss(
      model_fun=model_fun,
      use_cosine=FLAGS.semantic_similarity,
      embeddings=embeddings,
      interpolation=FLAGS.interpolation,
      kappa=FLAGS.kappa)
  output_difference = attack_setup.OutputDifference(model_fun)
  early_stopping_criterion = attack_setup.EarlyStopping(model_fun)

  dataset = tfds.load(FLAGS.dataset, split=FLAGS.split)
  batched_dataset = dataset.batch(FLAGS.batch_size)

  if FLAGS.sampling_strategy == 'uniform':
    sampling_strategy = sampling.uniform_sampling
  elif FLAGS.sampling_strategy == 'knn_euclidean':
    sampling_strategy = sampling.knn_sampling_euclidean
  else:
    sampling_strategy = sampling.knn_sampling_cosine

  optimizer = estimation.DiscreteZOO(sampling_strategy=sampling_strategy,
                                     embeddings=embeddings,
                                     adversarial_loss=adversarial_loss,
                                     num_to_sample=FLAGS.num_to_sample,
                                     reduce_mean=FLAGS.reduce_mean,
                                     descent=True,
                                     norm_embeddings=FLAGS.normalize_embeddings,
                                     vocab=vocab)

  with tf.io.gfile.GFile(FLAGS.output_file, 'w') as output_file:
    examples_attacked = 0
    total_successes = 0
    for batch in tqdm.tqdm(batched_dataset, desc='Attack Progress:'):
      if examples_attacked >= FLAGS.num_examples and FLAGS.num_examples != 0:
        break
      text_batch = batch['sentence'].numpy().tolist()
      # Tensorflow Datasets saves text as bytes.
      text_batch = [text.decode('utf-8') for text in text_batch]
      # Pre-process the batch of sentences into a numerical tensor.
      tokenized_text_batch = tokenize_sentences(text_batch)
      numericalized_batch = []
      for tokenized_text in tokenized_text_batch:
        numericalized_text = [
            token_to_id.get(token.lower(), FLAGS.oov_index)
            for token in tokenized_text
        ]
        numericalized_batch.append(numericalized_text)
      ragged_tensor_batch = tf.ragged.constant(numericalized_batch,
                                               dtype=tf.int32)
      tensor_batch = ragged_tensor_batch.to_tensor(FLAGS.padding_index)
      model_predicted_labels = tf.argmax(model_fun(tensor_batch),
                                         axis=-1,
                                         output_type=tf.int32)
      importance_scores = importance.scorer(tensor_batch, output_difference,
                                            importance.drop_position)
      adversarial_sentences, is_finished_attacks = attack_loop.loop(
          sentences=tensor_batch,
          labels=model_predicted_labels,
          optimizer=optimizer,
          token_importance_scores=importance_scores,
          early_stopping_criterion=early_stopping_criterion,
          iterations_per_token=FLAGS.changes_per_token,
          max_changes=FLAGS.token_changes_per_sentence)
      # Post-process the adversarial sentences back into detokenized text.
      adversarial_sentences_tokens = []
      for sentence in adversarial_sentences:
        sentence_tokens = [vocab[index] for index in sentence]
        adversarial_sentences_tokens.append(sentence_tokens)

      adversarial_sentences_strings = [
          detokenizer.detokenize(tokens)
          for tokens in adversarial_sentences_tokens
      ]
      adversarial_sentences_joined = '\n'.join(
          adversarial_sentences_strings) + '\n'
      output_file.write(adversarial_sentences_joined)
      total_successes += tf.reduce_sum(tf.cast(is_finished_attacks,
                                               tf.int32)).numpy()
      examples_attacked = examples_attacked + tensor_batch.shape[0]

    success_rate = total_successes / examples_attacked
    logging.info('Success Rate: %f', success_rate)


if __name__ == '__main__':
  app.run(main)