**This is not an officially supported Google product.**

# DiscreteZOO

This project is for generating adversarial examples against text classification
models.

## Adversarial Examples

In the image domain, we would generate adversarial examples by adding noise to
the pixel values in a targeted way to change the output label of the target
model. Unlike in images, in text we can't add noise directly to our input.
Instead we have to query discrete points, the tokens in our model's vocabulary.
We query the model with different replacement tokens to estimate a gradient in
the embedding space of the model. This estimated gradient points us towards an
area of space that increases the model's loss. Following this direction and then picking nearby tokens should allow us to eventually change the model's output
label.

## Algorithm

### Deciding which tokens to attack

Before we can attack the original sentence, we have to decide what order we want
to attack the tokens in. In order to do that, we loop over the tokens in the
target sentence, delete or mask the token, and compare the output probabilities
of the target model using the KL-Divergence. This gives us an importance score
per token in the sentence. Looping over the sentence and masking/deleting tokens
is found in `discretezoo/attack/importance.py`.

### Iterating over tokens in our attack order

Once we know which tokens are the most important, we want to attack them in
order of most important to least important. This happens inside
`discretezoo/attack/attack_loop.py`.

### Sampling tokens

Now that we're attacking a specific token, we will sample tokens either near
the current token in embedding space or uniformly over the vocabulary. These
two options are implemented in `discretezoo/attack/sampling.py` and used by the
gradient/displacement estimation code.

### Getting loss values

For each token that is sampled, we need to query the model and adversarial loss
with the sampled token in place of the original token. The loss value consists
of two parts, the adversarial loss and the semantic loss. Both are contained
in the directory `discretezoo/loss`. Adversarial loss is taken from the paper by
(Chen et al. 2017), ZOO: Zeroth Order Optimization based Black-box Attacks to
Deep Neural Networks without Training Substitute Models. The semantic loss is
either cosine similarity between the average token embeddings in the sentence,
the euclidean distance between the tokens, or with the cosine similarity between
Universal Sentence Encoder outputs (Cer et al. 2018).

### Estimating a gradient

We can estimate a slope as the difference in loss divided by the distance
between the two tokens in embedding space. Multiplying the slope by the
displacement between the two embeddings yields a gradient. Since this occurs for
every sampled token, the scaled displacements can either be summed up or
averaged to get a single estimate for the gradient of the loss with regards to
our target token. This occurs in `discretezoo/attack/estimation.py`

### Turning the gradient into a replacement token

Once we have our gradient (which is a vector from our current token and points
in the direction of increasing loss) it needs to be turned into a replacement
token. There are two methods implemented to do this. One method adds the
gradient to the embedding of the target token and then looks for nearby tokens
using cosine similarity. The other method selects the token with the maximum
inner product with the gradient, similar to hotflip. This too occurs in
`discretezoo/attack/estimation.py`.

### Helper functions and metrics

Helper functions used for calling the model, pre/post-precessing and tokenizing
text, loading embeddings, and combining our two losses are found in
`discretezoo/attack_setup.py`. BLEU score calculation and counting the number
of changed tokens is done in `discretezoo/metrics.py`.

## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

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
