import sys
import logging

from utils.general import get_current_time


class Logger(object):
    def __init__(self, log_filename: str = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

        if log_filename is not None:
            log_filename = f"./logs/{log_filename}.log"
        else:
            log_filename = f"./logs/log-{get_current_time()}.log"

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 文件流
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # 终端
        stdf = logging.StreamHandler(sys.stdout)
        stdf.setLevel(logging.INFO)
        stdf.setFormatter(formatter)
        self.logger.addHandler(stdf)

    def info(self, info):
        self.logger.info(info)
        sys.stdout.flush()



if __name__ == "__main__":
    L = Logger("12345")
    for i in range(10):
        L.info(f"{'-' * 20} Hello")
