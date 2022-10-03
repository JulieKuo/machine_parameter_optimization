import logging
from logging.handlers import TimedRotatingFileHandler


def set_log(filepath = "prog/logs/train.log", level = 3, freq = "W", interval = 4):
    format = '%(asctime)s %(levelname)s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    level_dict = {
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.ERROR,
        4: logging.WARNING,
        5: logging.CRITICAL,
    }

    fmt = logging.Formatter(format, datefmt)

    log_level = level_dict[level]

    logger = logging.getLogger()
    logger.setLevel(log_level)

    hdlr = TimedRotatingFileHandler(filename = filepath, when = freq, interval = interval, backupCount = 1)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)

    return logger