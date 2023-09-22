import os
import threading
from datetime import datetime, timedelta
from lib.const import META_DIR

UPDATE_TIME = META_DIR + 'update_time'
TIME_FORMAT = '%m/%d/%Y'
TIMER_INTERVAL = 24*60*60

class DSUpdater:
    def __init__(self, vector_store, datasources):
        self.vector_store = vector_store
        self.datasources = datasources
        os.makedirs(META_DIR, exist_ok=True)

    def _get_update_date(self):
        if os.path.exists(UPDATE_TIME):
            with open(UPDATE_TIME) as f:
                return datetime.strptime(f.readline(), TIME_FORMAT).date()
        return None

    def _save_next_update_date(self):
        now = datetime.now()
        with open(UPDATE_TIME, 'w+') as f:
            f.write(now.strftime(TIME_FORMAT))

    def _update_data(self):
        start_date = self._get_update_date()
        end_date = (datetime.now() + timedelta(1)).date()
        for ds_type, ds in self.datasources.items():
            for data in ds.get_update_data(start_date, end_date):
                self.vector_store.update(ds_type, data)
        self._save_next_update_date()
        threading.Timer(TIMER_INTERVAL, self._update_data)

    def initialize_data(self):
        update_date = self._get_update_date()
        for ds_type, ds in self.datasources.items():
            for data in ds.get_initial_data(update_date):
                self.vector_store.update(ds_type, data)
        self._save_next_update_date()
        threading.Timer(TIMER_INTERVAL, self._update_data)
