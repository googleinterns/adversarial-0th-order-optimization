from typing import List
import itertools

import sacrebleu


def changed_token_count(original_tokens: List[List[str]],
                        adversarial_tokens: List[List[str]]) -> List[int]:
  """Given two batches of lists of tokens, this finds per example changes.

  Arguments:
    original_tokens: A batch of lists containing the original sentences' tokens.
    adversarial_tokens: A batch of lists containing the adversarial sentences'
      tokens.

  Returns:
    A list containing how many tokens per example pair are unequal.
  """
  changed_token_counts = []
  for original_sentence, adversarial_sentence in zip(original_tokens,
                                                     adversarial_tokens):
    changed_token_counts.append(0)
    for original_token, adversarial_token in itertools.zip_longest(
        original_sentence, adversarial_sentence, fillvalue=''):
      if original_token != adversarial_token:
        changed_token_counts[-1] += 1
  return changed_token_counts


def sentence_bleu_scores(original_sentences: List[str],
                         adversarial_sentences: List[str]) -> List[int]:
  """This function iterates over sentences and calls sacrebleu to get scores.

  Arguments:
    original_sentences: A list of sentences before the adversarial attack.
    adversarial_sentences: A list of sentences after the adversarial attack.

  Returns:
    A list of BLEU scores comparing the adversarial sentences to the original
      sentences by using original sentences as references and adversarial
      sentences as hypotheses.
  """
  bleu_scores = []
  for original_sentence, adversarial_sentence in zip(original_sentences,
                                                     adversarial_sentences):
    score = sacrebleu.sentence_bleu(adversarial_sentence,
                                    original_sentence,
                                    use_effective_order=True)
    bleu_scores.append(score)
  return bleu_scores
