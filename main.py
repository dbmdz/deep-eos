import argparse

from eos import EOS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "tag", "extract"])
    parser.add_argument("--training-file",
                        help="Defines training data set")
    parser.add_argument("--test-file",
                        help="Defines test data set")
    parser.add_argument("--input-file",
                        help="Defines input file to be tagged")
    parser.add_argument("--epochs", default=5,
                        help="Defines number of training epochs")
    parser.add_argument(
        "--architecture",
        default="cnn",
        help="Neural network architectures, supported: cnn, lstm, bi-lstm, gru, bi-gru, mlp")
    parser.add_argument("--window-size", default=5,
                        help="Defines number of window size (char-ngram)")
    parser.add_argument("--batch-size", default=32,
                        help="Defines number of batch_size")
    parser.add_argument("--dropout", default=0.2,
                        help="Defines number dropout")
    parser.add_argument(
        "--min-freq",
        default=100,
        help="Defines the min. freq. a char must appear in data")
    parser.add_argument("--max-features", default=200,
                        help="Defines number of features for Embeddings layer")
    parser.add_argument("--embedding-size", default=128,
                        help="Defines Embeddings size")
    parser.add_argument("--kernel-size", default=8,
                        help="Defines Kernel size of CNN")
    parser.add_argument("--filters", default=6,
                        help="Defines number of filters of CNN")
    parser.add_argument("--pool-size", default=8,
                        help="Defines pool size of CNN")
    parser.add_argument("--hidden-dims", default=250,
                        help="Defines number of hidden dims")
    parser.add_argument("--strides", default=1,
                        help="Defines numer of strides for CNN")
    parser.add_argument("--lstm_gru_size", default=256,
                        help="Defines size of LSTM/GRU layer")
    parser.add_argument("--mlp-dense", default=6,
                        help="Defines number of dense layers for mlp")
    parser.add_argument("--mlp-dense-units", default=16,
                        help="Defines number of dense units for mlp")
    parser.add_argument("--model-filename", default='best_model.hdf5',
                        help="Defines model filename")
    parser.add_argument("--vocab-filename", default='vocab.dump',
                        help="Defines vocab filename")
    parser.add_argument("--eos-marker", default='</eos>',
                        help="Defines end-of-sentence marker used for tagging")
    args = parser.parse_args()

    nn_eos = NNEOS()

    if args.mode == "train":
        if not args.training_file:
            print("Training data file name is missing!")
            parser.print_help()
            exit(1)

        nn_eos.train(args.training_file,
                     str(args.architecture),
                     int(args.window_size),
                     int(args.epochs),
                     int(args.batch_size),
                     float(args.dropout),
                     int(args.min_freq),
                     int(args.max_features),
                     int(args.embedding_size),
                     int(args.lstm_gru_size),
                     int(args.mlp_dense),
                     int(args.mlp_dense_units),
                     int(args.kernel_size),
                     int(args.filters),
                     int(args.pool_size),
                     int(args.hidden_dims),
                     int(args.strides),
                     args.model_filename,
                     args.vocab_filename)
    elif args.mode == "test":
        if not args.test_file:
            print("Test data file name is missing!")
            parser.print_help()
            exit(1)

        nn_eos.test(args.test_file,
                    args.model_filename,
                    args.vocab_filename,
                    int(args.window_size),
                    int(args.batch_size))
    elif args.mode == "tag":
        if not args.input_file:
            print("Input file name is missing!")
            parser.print_help()
            exit(1)

        nn_eos.tag(args.input_file,
                   args.model_filename,
                   args.vocab_filename,
                   int(args.window_size),
                   int(args.batch_size),
                   args.eos_marker)

    elif args.mode == "extract":
        if not args.input_file:
            print("Input file name is missing!")
            parser.print_help()
            exit(1)

        nn_eos.extract(
            args.input_file, int(
                args.window_size), int(
                args.min_freq))


if __name__ == '__main__':
    parse_arguments()
