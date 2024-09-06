import argparse
import signal
import sys
import time
from .lib.datasources.ds_updater import DSUpdater
from .utils import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(args.config)
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
