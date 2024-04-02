from lib.model_manager.model_manager import ModelManager


class BaseContext:
    def __init__(self, config):
        self.model_manager = ModelManager(config)
