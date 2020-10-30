import datetime
import os
import csv

from absl import app
from absl import flags
from absl import logging
from nltk.tokenize import treebank
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from discretezoo import attack_setup
from discretezoo.attack import importance, attack_loop, estimation, sampling
from discretezoo.loss import semantic_similarity

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
flags.DEFINE_list(
    'special_tokens', [],
    'The index of vocabulary items that should not be generated. '
    'Must be integers.')
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
    'semantic_similarity', 'cosine', ['euclidean', 'cosine', 'use'],
    'This controls how similarity between two sentences is computed. '
    '"use" stands for Universal Sentence Encoder, a sentence embedding method '
    'and the resulting embeddings will be compared with cosine distance.')
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
# Logging
flags.DEFINE_string('output_file', None,
                    'The output file to write adversarial examples to.')
flags.DEFINE_string('tensorboard_logdir', None,
                    'The output directory to write tensorboard logs to.')
flags.DEFINE_string('tensorboard_profiling_dir', None,
                    'The directory to write profiling data to.')

flags.mark_flags_as_required(
    ['model', 'embeddings_file', 'vocab_file', 'dataset', 'output_file'])

TSV_HEADER = [
    'true_label', 'predicted_label', 'label_flipped', 'query_count',
    'original_sentence', 'adversarial_sentence'
]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.get_absl_handler().use_absl_log_file()
  if FLAGS.tensorboard_profiling_dir is not None:
    tf.profiler.experimental.start(FLAGS.tensorboard_profiling_dir)

  logging.info('Writing output to: %s', FLAGS.output_file)

  detokenizer = treebank.TreebankWordDetokenizer()

  if FLAGS.tensorboard_logdir:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_path = os.path.join(FLAGS.tensorboard_logdir, current_time)
    summary_writer = tf.summary.create_file_writer(tensorboard_path)
    logging.info('Writing tensorboard logs to %s', tensorboard_path)
  else:
    summary_writer = tf.summary.create_noop_writer()

  embeddings, token_to_id, vocab = attack_setup.load_embeddings(
      FLAGS.embeddings_file, FLAGS.vocab_file)

  tensorboard_logging = FLAGS.tensorboard_logdir is not None

  model_fun = attack_setup.ModelCallable(
      FLAGS.model,
      vocab,
      detokenizer.detokenize,
      include_tokenizer=FLAGS.include_tokenizer,
      padding_index=FLAGS.padding_index)

  if FLAGS.semantic_similarity == 'cosine':
    distance_fun = semantic_similarity.EmbeddedCosineDistance(embeddings)
  elif FLAGS.semantic_similarity == 'euclidean':
    distance_fun = semantic_similarity.EmbeddedEuclideanDistance(embeddings)
  else:
    distance_fun = semantic_similarity.UniversalSentenceEncoderDistance(
        detokenizer.detokenize, vocab, padding_index=FLAGS.padding_index)

  adversarial_loss = attack_setup.AdversarialLoss(
      model_fun=model_fun,
      distance_fun=distance_fun,
      embeddings=embeddings,
      interpolation=FLAGS.interpolation,
      kappa=FLAGS.kappa,
      tensorboard_logging=tensorboard_logging)
  output_difference = attack_setup.OutputDifference(model_fun)
  early_stopping_criterion = attack_setup.EarlyStopping(model_fun)

  dataset = tfds.load(FLAGS.dataset, split=FLAGS.split)
  sorted_dataset = attack_setup.sort_dataset(dataset)
  batched_dataset = sorted_dataset.batch(FLAGS.batch_size)

  if FLAGS.sampling_strategy == 'uniform':
    sampling_strategy = sampling.uniform_sampling
  elif FLAGS.sampling_strategy == 'knn_euclidean':
    sampling_strategy = sampling.knn_sampling_euclidean
  else:
    sampling_strategy = sampling.knn_sampling_cosine

  command_line_special_tokens = [int(index) for index in FLAGS.special_tokens]
  # This is to de-deduplicate any possible copies.
  special_tokens = ({FLAGS.padding_index,
                     FLAGS.oov_index}.union(command_line_special_tokens))
  special_tokens = list(special_tokens)

  optimizer = estimation.DiscreteZOO(sampling_strategy=sampling_strategy,
                                     embeddings=embeddings,
                                     adversarial_loss=adversarial_loss,
                                     num_to_sample=FLAGS.num_to_sample,
                                     reduce_mean=FLAGS.reduce_mean,
                                     descent=True,
                                     norm_embeddings=FLAGS.normalize_embeddings,
                                     vocab=vocab,
                                     special_tokens=special_tokens)

  with tf.io.gfile.GFile(FLAGS.output_file,
                         'w') as output_file, summary_writer.as_default():
    tsv_output = csv.writer(output_file,
                            delimiter='\t',
                            quoting=csv.QUOTE_NONE,
                            escapechar='\\')
    tsv_output.writerow(TSV_HEADER)
    examples_attacked = 0
    total_successes = 0
    for step, batch in enumerate(
        tqdm.tqdm(batched_dataset, desc='Attack Progress')):
      tf.summary.experimental.set_step(step)
      if examples_attacked >= FLAGS.num_examples and FLAGS.num_examples != 0:
        break
      text_batch = batch['sentence'].numpy().tolist()
      original_labels = batch['label']
      # Tensorflow saves text as bytes.
      decoded_text_batch = []
      for text in text_batch:
        decoded_text_batch.append([token.decode('utf-8') for token in text])
      decoded_text_batch_strings = [
          ' '.join(tokens) for tokens in decoded_text_batch
      ]
      # Log original tokenized texts.
      logging.debug("Original sentences: \n%s", decoded_text_batch)
      # Pre-process the batch of sentences into a numerical tensor.
      numericalized_batch = []
      for tokenized_text in decoded_text_batch:
        numericalized_text = [
            token_to_id.get(token.lower(), FLAGS.oov_index)
            for token in tokenized_text
        ]
        numericalized_batch.append(numericalized_text)
      ragged_tensor_batch = tf.ragged.constant(numericalized_batch,
                                               dtype=tf.int32)
      tensor_batch = ragged_tensor_batch.to_tensor(FLAGS.padding_index)
      model_fun.reset_query_tracking(tensor_batch)
      model_predicted_probabilities = model_fun(tensor_batch)
      model_predicted_labels = tf.argmax(model_predicted_probabilities,
                                         axis=-1,
                                         output_type=tf.int32)
      importance_scores = importance.scorer(tensor_batch,
                                            model_predicted_probabilities,
                                            output_difference,
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
      adversarial_sentences_strings = attack_setup.tensor_to_strings(
          adversarial_sentences, vocab, detokenizer.detokenize,
          FLAGS.padding_index)
      is_padding = adversarial_sentences == FLAGS.padding_index
      padding_per_sentence = tf.reduce_sum(tf.cast(is_padding, tf.int32),
                                           axis=-1)
      # TODO: Come up with a less hack-y way to ignore queries in importance
      # scoring for padding tokens.
      query_count = model_fun.query_count - padding_per_sentence
      query_count = query_count.numpy().tolist()
      tsv_data = zip(original_labels.numpy().tolist(),
                     model_predicted_labels.numpy().tolist(),
                     is_finished_attacks.numpy().tolist(), query_count,
                     decoded_text_batch_strings, adversarial_sentences_strings)
      tsv_output.writerows(tsv_data)

      total_successes += tf.reduce_sum(tf.cast(is_finished_attacks,
                                               tf.int32)).numpy()
      examples_attacked = examples_attacked + tensor_batch.shape[0]

    success_rate = total_successes / examples_attacked
    logging.info('Success Rate: %f', success_rate)
    if FLAGS.tensorboard_profiling_dir is not None:
      tf.profiler.experimental.stop()


if __name__ == '__main__':
  app.run(main)
