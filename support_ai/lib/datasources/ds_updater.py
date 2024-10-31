"""
This module provides the DSUpdater class for managing and updating data
from various data sources at specified intervals.
"""

import os
import threading
from datetime import datetime, timedelta

from support_ai.lib.const import META_DIR
from support_ai.lib.context import BaseContext
from support_ai.lib.vectorstore import VectorStore
from support_ai.lib.datasources.utils import get_datasources


UPDATE_TIME = META_DIR + 'update_time'
TIME_FORMAT = '%m/%d/%Y'
TIMER_INTERVAL = 24*60*60


class RepeatTimer(threading.Timer):
    """
    A timer that executes a function repeatedly at specified intervals.
    """

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class DSUpdater(BaseContext):
    """
    A class for updating data sources periodically.
    """

    def __init__(self, config):
        super().__init__(config)
        self.vector_store = VectorStore()
        self.datasources = get_datasources(config)
        self.update_timer = RepeatTimer(TIMER_INTERVAL, self.__trigger_update)
        self.update_thread = threading.Thread(target=self.__update_data_worker)
        self.stop_update_thread = threading.Event()
        self.update_cond = threading.Condition()
        os.makedirs(META_DIR, exist_ok=True)

    def __trigger_update(self):
        """
        Triggers an update by notifying the update condition.
        """
        with self.update_cond:
            self.update_cond.notify()

    def __get_update_date(self):
        """
        Retrieves the last update date from a file.

        Returns:
            datetime.date: The last update date, or None if the
                           update date file does not exist.
        """
        if os.path.exists(UPDATE_TIME):
            with open(UPDATE_TIME, encoding="utf-8") as f:
                return datetime.strptime(f.readline(), TIME_FORMAT).date()
        return None

    def __update_data(self):
        """
        Updates data from all data sources.
        """
        start_date = self.__get_update_date()
        end_date = (datetime.now() + timedelta(1)).date()
        for ds_type, ds in self.datasources.items():
            if self.stop_update_thread.is_set():
                return
            for data in ds.get_update_data(start_date, end_date):
                if self.stop_update_thread.is_set():
                    return
                self.vector_store.update(ds_type,
                                         ds.model.embeddings,
                                         data)
        self.__save_next_update_date()

    def __save_next_update_date(self):
        """
        Saves the current date as the next update date.
        """
        now = datetime.now()
        with open(UPDATE_TIME, 'w+', encoding="utf-8") as f:
            f.write(now.strftime(TIME_FORMAT))

    def __update_data_worker(self):
        """
        The worker method for updating data.
        """
        self.__update_data()
        while not self.stop_update_thread.is_set():
            with self.update_cond:
                self.update_cond.wait()
            self.__update_data()
        self.update_timer.cancel()

    def start_update_thread(self):
        """
        Starts the data update thread and timer.
        """
        self.stop_update_thread.clear()
        self.update_thread.start()
        self.update_timer.start()

    def cancel_update_thread(self):
        """
        Stops the data update thread and timer.
        """
        self.stop_update_thread.set()
        with self.update_cond:
            self.update_cond.notify()
        if self.update_thread.is_alive():
            self.update_thread.join()
