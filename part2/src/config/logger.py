import logging
from .config_helper import Configuration


class Logger:
    def __init__(self, name):
        self.logger_config = Configuration().get_config('logger')
        self.name = name
        self.log_format = self.logger_config['format']
        self.log_level = self.logger_config['level']

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        formatter = logging.Formatter(self.log_format)

        # Log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Log to file
        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info('\033[0;32m%s\033[0m' % message)

    def warning(self, message):
        self.logger.warning('\033[0;33m%s\033[0m' % message)

    def error(self, message):
        self.logger.error('\033[0;31m%s\033[0m' % message)

    def debug(self, message):
        self.logger.debug('\033[0;34m%s\033[0m' % message)