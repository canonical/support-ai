from . import const
from .context import BaseContext
from .datasources.ds_querier import DSQuerier
from .memory import Memory


class Chain(BaseContext):
    def __init__(self, config):
        super().__init__(config)
        if const.CONFIG_BASIC_MODEL not in config:
            raise ValueError(
                f'The config doesn\'t contain {const.CONFIG_BASIC_MODEL}')
        self.model = self.model_manager.get_model(
                        config[const.CONFIG_BASIC_MODEL])
        if const.CONFIG_MEMORY in config:
            self.memory = Memory(config[const.CONFIG_MEMORY],
                                 self.model.llm)
        else:
            self.memory = None

        self.ds_querier = DSQuerier(config)

    def __stream(self, output):
        delimiters = [' ', '\t', '\n']
        left, right = 0, 0
        for right in range(len(output)):
            if output[right] in delimiters:
                yield output[left:right + 1]
                left = right + 1
        if left < right:
            yield output[left:right + 1]

    def ask(self, query, ds_type=None, session=None):
        ds, doc = self.ds_querier.query(query, ds_type)
        content = ds.get_content(doc.metadata)
        if session is not None and self.memory is not None:
            content.summary = self.memory.integrate(session, query,
                                                    content.Summary)
        return self.__stream(ds.generate_output(content))

    def custom_api(self, ds_type, action, data):
        ds = self.ds_querier.get_ds(ds_type)
        content = ds.custom_api(action, data)
        return self.__stream(ds.generate_output(content))

    def clear_history(self, session):
        if self.memory is None:
            return
        self.memory.clear(session)
