import logging

class YamnetLogger():
    def __init__(self, file=None):
        self.logger = logging.getLogger("yamnet")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "{asctime} - {levelname} - {message}",
                style="{",
                datefmt="%Y-%m-%d %H:%M"
            )
            if file is not None:
                file_handler = logging.FileHandler(file, encoding="utf-8", mode="w")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)
        