"""eval: script that calculates various evaluation metrics."""

import click

EOS_TOKEN = "</eos>"

PUNCT_CHARS = [".", "!", "?", ":", ";"]


def read_file_to_token_list(file_name):
    """Read file, tokenize each line on spaces and return a list of tokens.

    :param file_name: File to be read in (default encoding is UTF-8)
    :return         : List of tokens
    """
    with open(file_name, mode='r', encoding='utf-8') as f_p:
        return [word for line in f_p for word in line.split()]


def evaluate(gold_tokens, system_tokens, verbose=False):
    """Evaluate F1-Score based on Gold standard and system output.

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

        if EOS_TOKEN in gold_token and EOS_TOKEN in system_token:
            tp_counter += 1
            output += "\ttp"
        elif EOS_TOKEN not in gold_token and gold_token.endswith(
                tuple(PUNCT_CHARS)) \
                and EOS_TOKEN not in system_token \
                and system_token.endswith(tuple(PUNCT_CHARS)):
            tn_counter += 1
            output += "\ttn"
        elif EOS_TOKEN in gold_token and EOS_TOKEN not in system_token:
            fn_counter += 1
            output += "\tfn"
        elif EOS_TOKEN not in gold_token and EOS_TOKEN in system_token:
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


@click.command()
@click.option('-g', '--gold', type=click.Path(exists=True), help='Gold standard')  # noqa: E501
@click.option('-s', '--system', type=click.Path(exists=True), help='System output')  # noqa: E501
@click.option('-v', '--verbose', count=True, help='Verbose output')
def parse_arguments(gold, system, verbose):
    """Parse commandline options."""
    gold_tokens = read_file_to_token_list(gold)
    system_tokens = read_file_to_token_list(system)

    precision, recall, f_score = evaluate(gold_tokens, system_tokens, verbose)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f_score)


if __name__ == '__main__':
    parse_arguments()  # pylint: disable=no-value-for-parameter
