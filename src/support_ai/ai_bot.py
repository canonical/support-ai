"""
Support-AI Command Line Tool
"""

import argparse
import uuid

from .lib.chain import Chain
from .utils import get_config


def parse_args():
    """
    Parses command-line arguments for the support-ai command line tool.

    Returns:
        argparse.Namespace: Parsed arguments, including the config file path.
    """
    parser = argparse.ArgumentParser(
        description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None,
                        help='Config path')
    return parser.parse_args()


def main():
    """
    Main function to execute the support-ai tool. Initializes configuration,
    session ID, and input loop for querying the support-ai chain.

    Prompts the user for input queries and processes each input through the
    chain. Exits if 'exit', 'quit', 'q', 'e', or 'x' is entered.
    """
    args = parse_args()
    config = get_config(args.config)
    chain = Chain(config)
    session = str(uuid.uuid4())

    while True:
        query = input(">")
        # transfer query to lower case
        query = query.lower().strip()
        if query in ['exit', 'quit', 'q', 'e', 'x']:
            break
        if not query:
            continue
        for token in chain.ask(query, session=session):
            print(token, end='')


if __name__ == '__main__':
    main()
