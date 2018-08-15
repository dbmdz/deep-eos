import argparse

eos_token = "</eos>"

punct_chars = [".", "!", "?", ":", ";"]


def read_file_to_token_list(file_name):
    """Reads file, tokenizes each line on spaces and returns a list of tokens

    :param file_name: File to be read in (default encoding is UTF-8)
    :return         : List of tokens
    """
    with open(file_name, mode='r', encoding='utf-8') as f:
        return [word for line in f for word in line.split()]


def evaluate(gold_tokens, system_tokens, verbose=False):
    """Evaluate F1-Score based on Gold standard and system output

    :param gold_tokens: List of tokens for gold standard
    :param system_tokens: List of tokens for system output
    :param verbose: switch on/off verbose output (default: False)
    :return: Tuple of Precision, Recall and F1-Score
    """

    tp_counter = 0
    tn_counter = 0
    fn_counter = 0
    fp_counter = 0

    for i, _ in enumerate(gold_tokens):
        gold_token = gold_tokens[i]
        system_token = system_tokens[i]

        output = gold_token + "\t" + system_token

        if eos_token in gold_token and eos_token in system_token:
            tp_counter += 1
            output += "\ttp"
        elif eos_token not in gold_token and gold_token.endswith(tuple(punct_chars)) \
                and eos_token not in system_token and system_token.endswith(tuple(punct_chars)):
            tn_counter += 1
            output += "\ttn"
        elif eos_token in gold_token and eos_token not in system_token:
            fn_counter += 1
            output += "\tfn"
        elif eos_token not in gold_token and eos_token in system_token:
            fp_counter += 1
            output += "\tfp"

        if verbose:
            print(output)
            if gold_token[0] != system_token[0]:
                print("Error at index", i)
                exit(1)

    precision = float(tp_counter / (tp_counter + fp_counter))
    recall = float(tp_counter / (tp_counter + fn_counter))

    f_score = float(2 * ((precision * recall) / (precision + recall)))

    return (precision, recall, f_score)


def parse_arguments():
    """Method for parsing commandline options"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold", help="Gold standard")
    parser.add_argument("-s", "--system", help="System output")
    parser.add_argument("-v", "--verbose", help="Verbose output",
                        action='store_true', default=False)
    args = parser.parse_args()

    if not args.gold:
        print("Gold standard file name is missing!")
        parser.print_help()
        exit(1)

    if not args.system:
        print("System output file name is missing!")
        parser.print_help()
        exit(1)

    gold_tokens = read_file_to_token_list(args.gold)
    system_tokens = read_file_to_token_list(args.system)

    precision, recall, f_score = evaluate(gold_tokens, system_tokens, args.verbose)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f_score)


if __name__ == '__main__':
    parse_arguments()
