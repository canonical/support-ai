"""
Chain Module for Processing Queries with Context and Memory
"""

from . import const
from .context import BaseContext
from .datasources.ds_querier import DSQuerier
from .memory import Memory


class Chain(BaseContext):
    """
    Manages queries, memory, and data sources to generate responses.
    """

    def __init__(self, config):
        """
        Initializes the Chain with configuration, model, memory, and
        data sources.

        Args:
            config: Configuration dictionary containing model and
                    memory settings.

        Raises:
            ValueError: If the basic model configuration is missing.
        """
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
        """
        Streams the output in segments based on delimiters.

        Args:
            output: The full output text to be streamed.

        Yields:
            str: A segment of the output split by delimiters.
        """
        delimiters = [' ', '\t', '\n']
        left = 0
        for right, c in enumerate(output):
            if c in delimiters:
                yield output[left:right + 1]
                left = right + 1
        if left < len(output):
            yield output[left:]

    def ask(self, query, ds_type=None, session=None):
        """
        Processes a query and returns generated responses with
        memory integration.

        Args:
            query: The user query to process.
            ds_type: Type of data source to query.
            session: Session identifier for memory context.

        Returns:
            generator: Streamed response segments.
        """
        ds, doc = self.ds_querier.query(query, ds_type)
        content = ds.get_content(doc.metadata)
        if session is not None and self.memory is not None:
            content.summary = self.memory.integrate(session, query,
                                                    content.Summary)
        return self.__stream(ds.generate_output(content))

    def custom_api(self, ds_type, action, data):
        """
        Calls a custom API action on a data source and returns the result.

        Args:
            ds_type: Data source type for the action.
            action: Action to execute on the data source.
            data: Data required by the API action.

        Returns:
            generator: Streamed response segments from the data source API.
        """
        ds = self.ds_querier.get_ds(ds_type)
        content = ds.custom_api(action, data)
        return self.__stream(ds.generate_output(content))

    def clear_history(self, session):
        """
        Clears the session history in memory.

        Args:
            session: Session identifier for which to clear history.
        """
        if self.memory is None:
            return
        self.memory.clear(session)
