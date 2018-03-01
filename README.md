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

We train a maximum of 10 epochs for each model. For the German and English
corpus (*Europarl*) the time per epoch is 55 minutes for the Bi-LSTM
model, 28 minutes for the LSTM model and 5 minutes for the CNN model. For each
language from the *SETimes* corpus the time per epoch is 5 minutes for the
Bi-LSMT model, 3 minutes for the LSTM model and 20 seconds for the CNN model.
Timings are performed on a server machine with a single Nvidia Tesla K20Xm and
Intel Xeon E5-2630.

The results on test set on the *SETimes* corpus are shown in the following table.
For each language the best neural network model
outperforms *OpenNLP*. On average, the best neural
network model is 0.38% better than *OpenNLP*. The worst neural network
model also outperforms *OpenNLP* for each language. On average, the worst
neural network model is 0.33% better than *OpenNLP*. In half of the
cases the bi-directional LSTM model is the best model. In almost all cases the
CNN model performs worse than the LSTM and bi-directional LSTM model, but it
still achieves better results than the *OpenNLP* model. This suggests that
the CNN model still needs more hyperparameter tuning.

| Language   | LSTM      | Bi-LSTM   | CNN       | *OpenNLP*
| ---------- | --------- | --------- | --------- | ---------
| German     | **97.59** | **97.59** | 97.50     | 97.38
| English    | 98.61     | **98.62** | 98.55     | 98.40
| Bulgarian  | 99.22     | **99.27** | 99.22     | 98.87
| Bosnian    | **99.58** | 99.52     | 99.53     | 99.25
| Greek      | 99.67     | **99.70** | 99.66     | 99.25
| Croatian   | **99.46** | 99.44     | 99.44     | 99.07
| Macedonian | 98.04     | **98.09** | 97.94     | 97.86
| Romanian   | 99.05     | 99.05     | **99.06** | 98.89
| Albanian   | **99.52** | 99.51     | 99.47     | 99.34
| Serbian    | 98.72     | **98.76** | 98.73     | 98.32
| Turkish    | 98.56     | **98.58** | 98.54     | 98.08

The first two rows in the table above show the results on
test set on the *Europarl* corpus. For both German and English the best
neural network model outperforms *OpenNLP*. The CNN
model performs worse than the LSTM and bi-directional LSTM model but still
achieves better results than *OpenNLP*. The bi-directional LSTM model is
the best model and achieves the best results for German and English. On
average, the best neural network model is 0.22% better than *OpenNLP*,
whereas the worst neural network model is still 0.14% better than
*OpenNLP*.

| Model     | Precision | Recall | F1
| --------- | --------- | ------ | ---------
| LSTM      | 56.62     | 96.25  | 71.29
| Bi-LSTM   | 60.00     | 97.50  | 74.29
| CNN       | 61.90     | 97.50  | **75.12**
| *OpenNLP* | 54.60     | 96.25  | 69.68

The table above shows the results for the *zero-shot* scenario.
The CNN model outperforms *OpenNLP* by a large margin and is 6% better
than *OpenNLP*. The CNN model also outperforms all other neural network
models. Interestingly, the CNN model performs better in a
*zero-shot* scenario than in the previous tasks (*Europarl* and
SETimes*). That suggests that the CNN model generalizes better than LSTM
or Bi-LSTM for unseen abbreviations. The worst neural network model (LSTM
model) still performs 1,6% better than *OpenNLP*.

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

# Contact (Bugs, Feedback, Contribution and more)

For questions about *deep-eos*, contact the current maintainer:
Stefan Schweter <stefan@schweter.it>. If you want to contribute to the project
please refer to the [Contributing](CONTRIBUTING.md) guide!

# License

To respect the Free Software Movement and the enormous work of Dr. Richard Stallman
this implementation is released under the *GNU Affero General Public License*
in version 3. More information can be found [here](https://www.gnu.org/licenses/licenses.html)
and in `COPYING`.
