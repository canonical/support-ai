import os
import threading
from datetime import datetime, timedelta

from ..const import META_DIR
from ..context import BaseContext
from ..vectorstore import VectorStore
from .utils import get_datasources


UPDATE_TIME = META_DIR + 'update_time'
TIME_FORMAT = '%m/%d/%Y'
TIMER_INTERVAL = 24*60*60


class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class DSUpdater(BaseContext):
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
        with self.update_cond:
            self.update_cond.notify()

    def __get_update_date(self):
        if os.path.exists(UPDATE_TIME):
            with open(UPDATE_TIME) as f:
                return datetime.strptime(f.readline(), TIME_FORMAT).date()
        return None

    def __update_data(self):
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
        now = datetime.now()
        with open(UPDATE_TIME, 'w+') as f:
            f.write(now.strftime(TIME_FORMAT))

    def __update_data_worker(self):
        self.__update_data()
        while not self.stop_update_thread.is_set():
            with self.update_cond:
                self.update_cond.wait()
            self.__update_data()
        self.update_timer.cancel()

    def start_update_thread(self):
        self.stop_update_thread.clear()
        self.update_thread.start()
        self.update_timer.start()

    def cancel_update_thread(self):
        self.stop_update_thread.set()
        with self.update_cond:
            self.update_cond.notify()
        if self.update_thread.is_alive():
            self.update_thread.join()
