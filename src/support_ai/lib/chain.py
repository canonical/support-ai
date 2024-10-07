from .const import CONFIG_BASIC_MODEL, CONFIG_MEMORY, CONFIG_SF
from .context import BaseContext
from .datasources.ds_querier import DSQuerier
from .memory import Memory


class Chain(BaseContext):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG_BASIC_MODEL not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_BASIC_MODEL}')
        self.model = self.model_manager.get_model(config[CONFIG_BASIC_MODEL])
        self.memory = Memory(config[CONFIG_MEMORY], self.model.llm) if CONFIG_MEMORY in config else None
        self.ds_querier = DSQuerier(config)

    def __stream(self, output):
        delimiters = [' ', '\t', '\n']
        l, r = 0, 0
        for r in range(len(output)):
            if output[r] in delimiters:
                yield output[l:r+1]
                l = r + 1
        if l < r:
            yield output[l:r+1]

    def ask(self, query, ds_type=None, session=None):
        ds, doc = self.ds_querier.query(query, ds_type)
        content = ds.get_content(doc.metadata)
        if session is not None and self.memory is not None:
            content.summary = self.memory.integrate(session, query, content.Summary)
        return self.__stream(ds.generate_output(content))

    def summarize_case(self, case_number):
        ds = self.ds_querier.get_ds(CONFIG_SF)
        return self.__stream(ds.summarize_case(case_number))

    def clear_history(self, session):
        if self.memory is None:
            return
        self.memory.clear(session)
