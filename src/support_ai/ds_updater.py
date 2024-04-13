import pkgutil
import signal
import sys
import time
import yaml
from .lib.const import CONFIG_FILE
from .lib.datasources.ds_updater import DSUpdater


def main():
    data = pkgutil.get_data(__package__, CONFIG_FILE)
    if data is None:
        raise Exception(f'{CONFIG_FILE} doesn\'t exist in {__package__}')
    config = yaml.safe_load(data.decode('utf-8'))
    ds_updater = DSUpdater(config)

    def signal_handler(*_):
        ds_updater.cancel_update_thread()
        sys.exit()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    ds_updater.start_update_thread()
    while True:
        time.sleep(30)


if __name__ == '__main__':
    main()
