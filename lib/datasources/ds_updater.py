import threading
from datetime import datetime, timedelta

TIMER_INTERVAL = 24*60*60

class DSUpdater:
    def __init__(self, vector_store, datasources):
        self.vector_store = vector_store
        self.datasources = datasources

    def start_update(self):
        now = datetime.now()
        start_time = now
        end_time = now + timedelta(1)
        for ds_type, ds in self.datasources.items():
            for data in ds.get_update_data(start_time, end_time):
                self.vector_store.update(ds_type, data)
        threading.Timer(TIMER_INTERVAL, self.start_update)
