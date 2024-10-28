import argparse
import uuid

from .lib.chain import Chain
from .utils import get_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None,
                        help='Config path')
    return parser.parse_args()


def main():
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
