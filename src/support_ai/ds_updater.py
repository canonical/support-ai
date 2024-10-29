"""
Support-AI Data Source Updater
"""

import argparse
import signal
import sys
import time

from .lib.datasources.ds_updater import DSUpdater
from .utils import get_config


def parse_args():
    """
    Parses command-line arguments for the support-ai data source updater.

    Returns:
        argparse.Namespace: Parsed arguments, including the config file path.
    """
    parser = argparse.ArgumentParser(
        description='Command line tool for support-ai')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    return parser.parse_args()


def main():
    """
    Main function to initialize and start the data source updater.

    Loads the configuration, initializes the DSUpdater, and sets up signal
    handling for graceful termination. Runs the update thread continuously
    until a termination signal is received.
    """
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
