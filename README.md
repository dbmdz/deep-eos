# General-Purpose Neural Networks for Sentence Boundary Detection

In this repository we present general-purpose neural network models for sentence
boundary detection.
We report on a series of experiments with long short-term memory (LSTM),
bidirectional long short-term memory (Bi-LSTM) and convolutional neural network
(CNN) for sentence boundary detection. We show that these neural networks
architectures achieve state-of-the-art results both on multi-lingual benchmarks
and on a *zero-shot* scenario.

## Introduction

The task of sentence boundary detection is to identify sentences within a text.
Many natural language processing tasks take a sentence as an input unit, such
as part-of-speech tagging
([Manning, 2011](http://dl.acm.org/citation.cfm?id=1964799.1964816)), dependency
parsing ([Yu and Vu, 2017](http://aclweb.org/anthology/P17-2106)), named entity
recognition or machine translation.

Sentence boundary detection is a nontrivial task, because of the ambiguity of
the period sign `.`, which has several functions ([Grefenstette and
Tapanainen, 1994](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.28.5162)), e.g.:

* End of sentence
* Abbreviation
* Acronyms and initialism
* Mathematical numbers

A sentence boundary detection system has to resolve the use of ambiguous
punctuation characters to determine if the punctuation character is a true
end-of-sentence marker. In this implementation we define `?!:;.` as potential
end-of sentence markers.

Various approaches have been employed to achieve sentence boundary detection in
different languages. Recent research in sentence boundary detection focus on
machine learning techniques,
such as hidden Markov models ([Mikheev, 2002](http://dx.doi.org/10.1162/089120102760275992)),
maximum entropy ([Reynar and Ratnaparkhi, 1997](https://doi.org/10.3115/974557.974561)),
conditional random fields ([Tomanek et al., 2007](http://www.bootstrep.org/files/publications/FSU_2007_Tomanek_Pacling.pdf)),
decision tree ([Wong et al., 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4030568/))
and neural networks ([Palmer and Hearst, 1997](http://dl.acm.org/citation.cfm?id=972695.972697)).
[Kiss and Strunk (2006)](http://dx.doi.org/10.1162/coli.2006.32.4.485) use an
unsupervised sentence detection system called Punkt, which does not depend on
any additional resources. The system use collocation information as evidence
from unannotated corpora to detect e.g. abbreviations or ordinal numbers.

The sentence boundary detection task can be treated as a classification
problem. Our work is similar to the *SATZ* system, proposed by
[Palmer and Hearst (1997)](http://dl.acm.org/citation.cfm?id=972695.972697),
which uses a fully-connected feed-forward neural network. The *SATZ* system
disambiguates a punctuation mark given a context of *k* surrounding words.
This is different to our approach, as we use a char-based context window instead
of a word-based context window.

In the present work, we train different architectures of neural
networks, such as long short-term memory (LSTM), bidirectional long short-term
memory (Bi-LSTM) and convolutional neural network (CNN) and compare the results
with *OpenNLP*. *OpenNLP* is a state-of-the-art tool and uses a maximum
entropy model for sentence boundary detection. To test the robustness of our
models, we use the *Europarl* corpus for German and English and the
*SETimes* corpus for nine different Balkan languages.

Additionally, we use a *zero-shot* scenario to test our model on unseen
abbreviations. We show that our models outperform *OpenNLP* both for each
language and on the zero-shot learning task. Therefore, we conclude that our
trained models can be used for building a robust, language-independent
state-of-the-art sentence boundary detection system.

# Datasets

Similar to [Wong et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4030568/)
we use the *Europarl* corpus ([Koehn, 2005](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf))
for our experiments. The *Europarl* parallel corpus is extracted from the
proceedings of the European Parliament and is originally created for the
research of statistical machine translation systems. We only use German and English from
*Europarl*. [Wong et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4030568/) do not mention
that the *Europarl* corpus is not fully sentence-segmented. The
*Europarl* corpus has a one-sentence per line data format. Unfortunately,
in some cases one or more sentences appear in a line. Thus, we define the
*Europarl* corpus as "quasi"-sentence segmented corpus.

We use the *SETimes* corpus ([Tyers and Alperen, 2010](http://xixona.dlsi.ua.es/~fran/publications/lrec2010.pdf))
as a second corpus for
our experiments. The *SETimes* corpus is based on the content published on
the *SETimes.com news portal* and contains parallel texts in ten
languages. Aside from English the languages contained in the *SETimes*
corpus fall into several linguistic groups: Turkic (Turkish), Slavic
(Bulgarian, Croatian, Macedonian and Serbian), Hellenic (Greek), Romance
(Romanian) and Albanic (Albanian). The *SETimes* corpus is also a
"quasi"-sentence segmented corpus. For our experiments we use all the
mentioned languages except English, as we use an English corpus from
*Europarl*.  We do not use any additional data like abbreviation lists.

For a *zero-shot* scenario we extracted 80 German abbreviations including
their context in a sentence from Wikipedia. These abbreviations do not exist in
the German *Europarl* corpus.

## Preprocessing

Both *Europarl* and *SETimes* are not tokenized. Text tokenization
(or, equivalently, segmentation) is highly non-trivial for many
languages ([Schütze, 2017](http://aclanthology.info/papers/E17-1074/nonsymbolic-text-representation)).
It is problematic even for English as word
tokenizers are either manually designed or trained. For our proposed sentence
boundary detection system we use a similar idea from [Lee et al. (2016)](https://arxiv.org/abs/1610.03017).
They use a character-based approach
without explicit segmentation for neural machine translation. We also use a
character-based context window, so no explicit segmentation of input text is
necessary.

For both corpora we use the following preprocessing steps: (a) we
remove duplicate sentences, (b) we extract only sentences with ends with a
potential end-of-sentence marker.  For *Europarl* and *SETimes* each
text for a language is split into train, dev and test sets. The following table
shows a detailed summary of the training, development
and test sets used for each language.

| Language   | # Train   | # Dev   | # Test
| ---------- | --------- | ------- | -------
| German     | 1,476,653 | 184,580 | 184,580
| English    | 1,474,819 | 184,352 | 184,351
| Bulgarian  | 148,919   | 18,615  | 18,614
| Bosnian    | 97,080    | 12,135  | 12,134
| Greek      | 159,000   | 19,875  | 19,874
| Croatian   | 143,817   | 17,977  | 17,976
| Macedonian | 144,631   | 18,079  | 18,078
| Romanian   | 148,924   | 18,615  | 18,615
| Albanian   | 159,323   | 19,915  | 19,915
| Serbian    | 158,507   | 19,813  | 19,812
| Turkish    | 144,585   | 18,073  | 18,072

## Download

A script for automatically downloading and extracting the datasets is available
and can be used with:

```bash
./download_data.sh
```

Training, development and testdata is located in the `data` folder.

# Model

We use three different architectures of neural networks: long short-term memory
(LSTM), bidirectional long short-term memory (Bi-LSTM) and convolutional neural
network (CNN). All three models capture information at the character level. Our
models disambiguate potential end-of-sentence markers followed by a whitespace
or line break given a context of *k* surrounding characters. The potential
end-of-sentence marker is also included in the context window. The following
table shows an example of a sentence and its extracted
contexts: left context, middle context and right context. We also include the
whitespace or line break after a potential end-of-sentence marker.


| Input sentence        | Left  | Middle | Right
| --------------------- | ----- | ------ | -----
| I go to Mr. Pete Tong | to Mr | .      | _Pete

## LSTM

We use a standard
LSTM ([Hochreiter and Schmidhuber, 1997](http://dx.doi.org/10.1162/neco.1997.9.8.1735);
[Gers et al., 2000](http://dx.doi.org/10.1162/089976600300015015))
network with an embedding size of 128. The number of hidden states is 256. We
apply dropout with probability of 0.2 after the hidden layer during training.
We apply a sigmoid non-linearity before the prediction layer.

## Bi-LSTM

Our bidirectional LSTM network uses an embedding size of 128 and 256 hidden
states. We apply dropout with a probability of 0.2 after the hidden layer
during training, and we apply a sigmoid non-linearity before the prediction
layer.

## CNN

For the convolutional neural network we use a 1D convolution layer with 6
filters and a stride size of 1 ([Waibel et al., 1989](http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf)).
The output of the
convolution filter is fed through a global max pooling layer and the pooling
output is concatenated to represent the context. We apply one 250-dimensional
hidden layer with ReLU non-linearity before the prediction layer. We apply
dropout with a probability of 0.2 during training.

## Other Hyperparameters

Our proposed character-based model disambiguates a punctuation mark given a
context of *k* surrounding characters. In our experiments we found that a
context size of 5 surrounding characters gives the best results. We found that
it is very important to include the end-of-sentence marker in the context, as
this increases the F1-score of 2%.  All models are trained with averaged
stochastic gradient descent with a learning rate of 0.001 and mini-batch size
of 32. We use Adam for first-order gradient-based optimization. We use binary
cross-entropy as loss function. We do not tune hyperparameters for each
language. Instead, we tune hyperparameters for one language (English) and use
them across languages. The following table shows the number of
trainable parameters for each model.

| Model   | # Parameters
| ------- | ------------
| LSTM    | 420,097
| Bi-LSTM | 814,593
| CNN     | 33,751

# Results

We train a maximum of 5 epochs for each model. For the German and English
corpus (*Europarl*) the time per epoch is 54 minutes for the Bi-LSTM
model, 35 minutes for the LSTM model and 7 minutes for the CNN model. For each
language from the *SETimes* corpus the time per epoch is 6 minutes for the
Bi-LSTM model, 4 minutes for the LSTM model and 50 seconds for the CNN model.
Timings are performed on a *DGX-1* with a Nvidia *P-100*.

## Development set

The results on the development set for both *Europarl* and *SETimes* are shown
in the following table. Download link for model and vocab files for each
language are included, as well as detailed evaluation results.

| Language   | LSTM                                                                                                                                                                                                                                                                                                     | Bi-LSTM                                                                                                                                                                                                                                                                                                           | CNN                                                                                                                                                                                                                                                                                               | *OpenNLP*
| -----------| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
| German     | [0.9759](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.dev.lstm.evaluation)     ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-de.vocab)) | [**0.9760**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.dev.bi-lstm.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-de.vocab)) | [0.9751](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.dev.cnn.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-de.vocab)) | 0.9736
| English    | [**0.9864**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.dev.lstm.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-en.vocab)) | [0.9863](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.dev.bi-lstm.evaluation)     ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-en.vocab)) | [0.9861](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.dev.cnn.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-en.vocab)) | 0.9843
| Bulgarian  | [**0.9928**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.dev.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bg.vocab)) | [0.9926](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.dev.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bg.vocab)) | [0.9924](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bg.vocab)) | 0.9900
| Bosnian    | [0.9953](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.dev.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bs.vocab)) | [**0.9958**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bs.vocab)) | [0.9952](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bs.vocab)) | 0.9921
| Greek      | [0.9959](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.dev.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-el.vocab)) | [**0.9964**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-el.vocab)) | [0.9959](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-el.vocab)) | 0.9911
| Croatian   | [0.9947](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.dev.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-hr.vocab)) | [**0.9948**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-hr.vocab)) | [0.9946](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-hr.vocab)) | 0.9917
| Macedonian | [0.9795](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.dev.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-mk.vocab)) | [**0.9799**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-mk.vocab)) | [0.9794](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-mk.vocab)) | 0.9776
| Romanian   | [**0.9906**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.dev.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-ro.vocab)) | [0.9904](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.dev.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-ro.vocab)) | [0.9903](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-ro.vocab)) | 0.9888
| Albanian   | [**0.9954**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.dev.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sq.vocab)) | [**0.9954**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sq.vocab)) | [0.9945](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sq.vocab)) | 0.9934
| Serbian    | [**0.9891**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.dev.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sr.vocab)) | [0.9890](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.dev.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sr.vocab)) | [0.9886](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sr.vocab)) | 0.9838
| Turkish    | [0.9860](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.dev.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-tr.vocab)) | [**0.9867**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.dev.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-tr.vocab)) | [0.9858](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.dev.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-tr.vocab)) | 0.9830

For each language the best neural network model outperforms *OpenNLP*. On
average, the best neural network model is 0.32% better than *OpenNLP*. The worst
neural network model also outperforms *OpenNLP* for each language. On average,
the worst neural network model is 0.26% better than *OpenNLP*. In over 60% of
the cases the bi-directional LSTM model is the best model. In almost all cases
the CNN model performs worse than the LSTM and bi-directional LSTM model, but
it still achieves better results than the *OpenNLP* model. This suggests that
the CNN model still needs more hyperparameter tuning.

## Test set

The results on the development set for both *Europarl* and *SETimes* are shown
in the following table. Download link for model and vocab files for each
language are included, as well as detailed evaluation results.

| Language   | LSTM                                                                                                                                                                                                                                                                                                      | Bi-LSTM                                                                                                                                                                                                                                                                                                            | CNN                                                                                                                                                                                                                                                                                                 | *OpenNLP*
| -----------| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
| German     | [0.975](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.test.lstm.evaluation)      ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-de.vocab)) | [**0.9760**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.test.bi-lstm.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-de.vocab)) | [0.9751](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.de.sentences.test.cnn.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-de.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-de.vocab))  | 0.9738
| English    | [**0.9861**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.test.lstm.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-en.vocab)) | [0.9860](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.test.bi-lstm.evaluation)     ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-en.vocab)) | [0.9858](https://github.com/stefan-it/deep-eos/releases/download/v0.1/europarl-v7.de-en.en.sentences.test.cnn.evaluation) ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-en.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-en.vocab))  | 0.9840
| Bulgarian  | [0.9922](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.test.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bg.vocab)) | [**0.9923**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.test.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bg.vocab)) | [0.9919](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bg-en.bg.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bg.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bg.vocab))  | 0.9887
| Bosnian    | [0.9957](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.test.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-bs.vocab)) | [**0.9959**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.test.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-bs.vocab)) | [0.9953](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.bs-en.bs.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bs.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-bs.vocab))  | 0.9925
| Greek      | [0.9967](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.test.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-el.vocab)) | [**0.9969**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.test.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-el.vocab)) | [0.9963](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.el-en.el.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-el.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-el.vocab))  | 0.9925
| Croatian   | [0.9946](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.test.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-hr.vocab)) | [**0.9948**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.test.bi-lstm.evaluation)     ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-hr.vocab)) | [0.9943](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-hr.hr.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-hr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-hr.vocab)) | 0.9907
| Macedonian | [0.9810](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.test.lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-mk.vocab)) | [**0.9811**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.test.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-mk.vocab)) | [0.9794](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-mk.mk.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-mk.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-mk.vocab))  | 0.9786
| Romanian   | [**0.9907**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.test.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-ro.vocab)) | [0.9906](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.test.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-ro.vocab)) | [0.9904](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-ro.ro.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-ro.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-ro.vocab))  | 0.9889
| Albanian   | [**0.9953**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.test.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sq.vocab)) | [0.9949](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.test.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sq.vocab)) | [0.9940](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sq.sq.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sq.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sq.vocab))  | 0.9934
| Serbian    | [**0.9877**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.test.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-sr.vocab)) | [**0.9877**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.test.bi-lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-sr.vocab)) | [0.9870](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-sr.sr.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-sr.vocab))  | 0.9832
| Turkish    | [**0.9858**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.test.lstm.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/lstm-tr.vocab)) | [0.9854](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.test.bi-lstm.evaluation)        ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/bi-lstm-tr.vocab)) | [0.9854](https://github.com/stefan-it/deep-eos/releases/download/v0.1/SETIMES2.en-tr.tr.sentences.test.cnn.evaluation)    ([model](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-tr.model), [vocab](https://github.com/stefan-it/deep-eos/releases/download/v0.1/cnn-tr.vocab))  | 0.9808

For each language the best neural network model
outperforms *OpenNLP*. On average, the best neural
network model is 0.32% better than *OpenNLP*. The worst neural network
model also outperforms *OpenNLP* for each language. On average, the worst
neural network model is 0.25% better than *OpenNLP*. In half of the
cases the bi-directional LSTM model is the best model. In almost all cases the
CNN model performs worse than the LSTM and bi-directional LSTM model, but it
still achieves better results than the *OpenNLP* model.

## *Zero-shot*

| Model     | Precision | Recall | F1-Score
| --------- | --------- | ------ | ------------------------------------------------------------------------------------------------------
| LSTM      | 0.6046    | 0.9750 | [0.7464](https://github.com/stefan-it/deep-eos/releases/download/v0.1/zeroshot.lstm.evaluation)
| Bi-LSTM   | 0.6341    | 0.9750 | [**0.7684**](https://github.com/stefan-it/deep-eos/releases/download/v0.1/zeroshot.bi-lstm.evaluation)
| CNN       | 0.57350   | 0.9750 | [0.7222](https://github.com/stefan-it/deep-eos/releases/download/v0.1/zeroshot.cnn.evaluation)
| *OpenNLP* | 54.60     | 96.25  | 69.68

The table above shows the results for the *zero-shot* scenario. The
bi-directional LSTM model outperforms *OpenNLP* by a large margin and is 7%
better than *OpenNLP*. The bi-directional LSTM model also outperforms all other
neural network models. That suggests that the bi-directional LSTM model
generalizes better than LSTM or CNN for unseen abbreviations. The worst neural
network model (CNN) still performs 2,5% better than *OpenNLP*.

# Conclusion

In this repository, we propose a general-purpose system for sentence boundary
detection using different architectures of neural networks. We use the
*Europarl* and *SETimes* corpus and compare our proposed models with
*OpenNLP*. We achieve state-of-the-art results.

In a *zero-shot* scenario, in which no manifestation of the test
abbreviations is observed during training, our system is also robust against
unseen abbreviations.

The fact that our proposed neural network models perform well on different
languages and on a *zero-shot* scenario leads us to the conclusion that our
system is a *general-purpose* system.

## Evaluation

To reproduce this results, the following scripts can be used:

* `benchmark_all.sh` - runs evaluation for various neural network models and
  all languages
* `benchmark_all_opennlp` - runs evaluation for *OpenNLP* for all languages

# Implementation

We use *Keras* and *TensorFlow* for the implementation of the neural network
architectures.

## Options

The following commandline options are available:

```bash
$ python3 main.py --help
usage: main.py [-h] [--training-file TRAINING_FILE] [--test-file TEST_FILE]
               [--input-file INPUT_FILE] [--epochs EPOCHS]
               [--architecture ARCHITECTURE] [--window-size WINDOW_SIZE]
               [--batch-size BATCH_SIZE] [--dropout DROPOUT]
               [--min-freq MIN_FREQ] [--max-features MAX_FEATURES]
               [--embedding-size EMBEDDING_SIZE] [--kernel-size KERNEL_SIZE]
               [--filters FILTERS] [--pool-size POOL_SIZE]
               [--hidden-dims HIDDEN_DIMS] [--strides STRIDES]
               [--lstm_gru_size LSTM_GRU_SIZE] [--mlp-dense MLP_DENSE]
               [--mlp-dense-units MLP_DENSE_UNITS]
               [--model-filename MODEL_FILENAME]
               [--vocab-filename VOCAB_FILENAME] [--eos-marker EOS_MARKER]
               {train,test,tag,extract}

positional arguments:
  {train,test,tag,extract}

optional arguments:
  -h, --help            show this help message and exit
  --training-file TRAINING_FILE
                        Defines training data set
  --test-file TEST_FILE
                        Defines test data set
  --input-file INPUT_FILE
                        Defines input file to be tagged
  --epochs EPOCHS       Defines number of training epochs
  --architecture ARCHITECTURE
                        Neural network architectures, supported: cnn, lstm,
                        bi-lstm, gru, bi-gru, mlp
  --window-size WINDOW_SIZE
                        Defines number of window size (char-ngram)
  --batch-size BATCH_SIZE
                        Defines number of batch_size
  --dropout DROPOUT     Defines number dropout
  --min-freq MIN_FREQ   Defines the min. freq. a char must appear in data
  --max-features MAX_FEATURES
                        Defines number of features for Embeddings layer
  --embedding-size EMBEDDING_SIZE
                        Defines Embeddings size
  --kernel-size KERNEL_SIZE
                        Defines Kernel size of CNN
  --filters FILTERS     Defines number of filters of CNN
  --pool-size POOL_SIZE
                        Defines pool size of CNN
  --hidden-dims HIDDEN_DIMS
                        Defines number of hidden dims
  --strides STRIDES     Defines numer of strides for CNN
  --lstm_gru_size LSTM_GRU_SIZE
                        Defines size of LSTM/GRU layer
  --mlp-dense MLP_DENSE
                        Defines number of dense layers for mlp
  --mlp-dense-units MLP_DENSE_UNITS
                        Defines number of dense units for mlp
  --model-filename MODEL_FILENAME
                        Defines model filename
  --vocab-filename VOCAB_FILENAME
                        Defines vocab filename
  --eos-marker EOS_MARKER
                        Defines end-of-sentence marker used for tagging
```

## Training

A new model can be trained using the `train` parameter. The only mandatory
argument in training mode is the `--training-file` parameter. This parameter
specifices the training file with sentence-separated entries.

```bash
python3 main.py train --training-file <TRAINING_FILE>
```

## Testing

A previously trained model can be evaluated using the `test` parameter. The only
mandatory argument for the testing mode is the `--test-file` parameter, that
specifies the test file with sentence-separated entries.

```bash
python3 main.py test --test-file <TEST_FILE>
```

## Tagging

To tag an input text with a previously trained model, the `tag` parameter must
be used in combination with specifying the to be tagged input text via the
`--input-file` parameter.

```bash
python3 main.py tag --input-file INPUT_FILE
```

## Evaluation

A evaluation script can be found in the `eos-eval` folder. The main arguments
for the `eval.py` script are:

```bash
$ python3 eval.py --help
usage: eval.py [-h] [-g GOLD] [-s SYSTEM] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD, --gold GOLD  Gold standard
  -s SYSTEM, --system SYSTEM
                        System output
  -v, --verbose         Verbose outpu
```

The system and gold standard file must use `</eos>` as end-of-sentence marker.
Then the evaluations script calculates precision, recall and F1-score. The
`--verbose` parameter gives a detailed output of e.g. *false negatives*.

# Acknowledgments

We would like to thank the *Leibniz-Rechenzentrum der Bayerischen Akademie der
Wissenschaften* ([LRZ](https://www.lrz.de/english/)) for giving us access to the
NVIDIA *DGX-1* supercomputer.

# Contact (Bugs, Feedback, Contribution and more)

For questions about *deep-eos*, contact the current maintainer:
Stefan Schweter <stefan@schweter.it>. If you want to contribute to the project
please refer to the [Contributing](CONTRIBUTING.md) guide!

# License

To respect the Free Software Movement and the enormous work of Dr. Richard Stallman
this implementation is released under the *GNU Affero General Public License*
in version 3. More information can be found [here](https://www.gnu.org/licenses/licenses.html)
and in `COPYING`.
