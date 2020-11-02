"""Tests for the file discretezoo_main.py"""
from absl.testing import absltest
import numpy as np

from discretezoo import metrics


class MetricsTest(absltest.TestCase):
  """This class contains the test cases for the functions in metrics.py"""

  def test_changed_token_count(self):
    """This function tests the function changed_token_count."""
    original_sentences = [['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'],
                          ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'],
                          ['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e']]
    adversarial_sentences = [['a', 'b', 'c', 'd',
                              'e'], ['f', 'b', 'c', 'd', 'e'],
                             ['f', 'g', 'c', 'd',
                              'e'], ['f', 'g', 'h', 'd', 'e'],
                             ['f', 'g', 'h', 'j', 'e'],
                             ['f', 'g', 'h', 'j', 'k']]

    expected_changed_tokens = [0, 1, 2, 3, 4, 5]
    test_changed_tokens = metrics.changed_token_count(original_sentences,
                                                      adversarial_sentences)

    self.assertListEqual(expected_changed_tokens, test_changed_tokens)

  def test_bleu_scores(self):
    """This function tests the function sentence_bleu_scores."""
    original_sentences = ['a b c d e f g', 'a b c d e f g']
    adversarial_sentences = ['a b c h e f g', 'a b c d e f g']
    expected_bleu_scores = [48.892302243490086, 100.0]
    test_bleu_scores = metrics.sentence_bleu_scores(original_sentences,
                                                    adversarial_sentences)
    np.testing.assert_allclose(expected_bleu_scores, test_bleu_scores)


if __name__ == '__main__':
  absltest.main()
