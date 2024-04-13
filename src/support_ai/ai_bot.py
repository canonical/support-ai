import pkgutil
import uuid
import yaml
from .lib.const import CONFIG_FILE
from .lib.chain import Chain


def main():
    data = pkgutil.get_data(__package__, CONFIG_FILE)
    if data is None:
        raise Exception(f'{CONFIG_FILE} doesn\'t exist in {__package__}')
    config = yaml.safe_load(data.decode('utf-8'))
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
