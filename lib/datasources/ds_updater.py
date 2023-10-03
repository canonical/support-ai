import os
import threading
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from lib.const import META_DIR

UPDATE_TIME = META_DIR + 'update_time'
TIME_FORMAT = '%m/%d/%Y'
TIMER_INTERVAL = 24*60*60
SYMPTOM_PROMPT = """Generate five symptoms of the following:
    "{context}"
    CONCISE SYMPTOMS:"""

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class DSUpdater:
    def __init__(self, llm, vector_store, datasources):
        self.llm = llm
        self.vector_store = vector_store
        self.datasources = datasources
        self.update_timer = RepeatTimer(TIMER_INTERVAL, self._trigger_update)
        self.update_thread = threading.Thread(target=self._update_data)
        self.stop_update_thread = threading.Event()
        self.update_cond = threading.Condition()
        os.makedirs(META_DIR, exist_ok=True)

    def _trigger_update(self):
        self.update_cond.acquire()
        self.update_cond.notify()
        self.update_cond.release()

    def _get_update_date(self):
        if os.path.exists(UPDATE_TIME):
            with open(UPDATE_TIME) as f:
                return datetime.strptime(f.readline(), TIME_FORMAT).date()
        return None

    def _save_next_update_date(self):
        now = datetime.now()
        with open(UPDATE_TIME, 'w+') as f:
            f.write(now.strftime(TIME_FORMAT))

    def _replace_invalid_token_in_collection(self, collection):
        return collection.replace(' ', '_')

    def _generate_symptoms(self, doc):
        prompt = PromptTemplate.from_template(SYMPTOM_PROMPT)
        query = prompt.format_prompt(context=doc)
        return self.llm.llm(query.to_string())

    def _parse_data(self, data):
        data.Collection = self._replace_invalid_token_in_collection(data.Collection)
        data.Document = self._generate_symptoms(data.Document)
        return data

    def _update_data(self):
        while True:
            self.update_cond.acquire()
            self.update_cond.wait()

            if self.stop_update_thread.is_set():
                break

            start_date = self._get_update_date()
            end_date = (datetime.now() + timedelta(1)).date()
            for ds_type, ds in self.datasources.items():
                for data in ds.get_update_data(start_date, end_date):
                    self.vector_store.update(ds_type, self._parse_data(data))
            self._save_next_update_date()
            self.update_cond.release()
        self.update_timer.cancel()
        self.update_cond.notify()
        self.update_cond.release()

    def start_update_thread(self):
        self.stop_update_thread.clear()
        self.update_thread.start()
        self.update_timer.start()

    def cancel_update_thread(self):
        self.stop_update_thread.set()
        self.update_cond.acquire()
        self.update_cond.notify()
        self.update_cond.wait()
        self.update_cond.release()

    def initialize_data(self):
        update_date = self._get_update_date()
        for ds_type, ds in self.datasources.items():
            for data in ds.get_initial_data(update_date):
                self.vector_store.update(ds_type, self._parse_data(data))
        self._save_next_update_date()
