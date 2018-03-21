import sys
import datetime
import os


class Logger(object):
    def __init__(self, output_dir):
        ensure_dir(output_dir)

        filename = str(datetime.datetime.now())
        self.log_dir = os.path.join(output_dir, filename)
        ensure_dir(self.log_dir)
        file_path = os.path.join(self.log_dir, 'log.log')
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        if message == '\n':
            return
        time = datetime.datetime.now()
        message = f'{time}: {message}\n'
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


OUTPUT_DIR = 'logs'


def init_log():
    logger = Logger(OUTPUT_DIR)
    sys.stdout = logger
    sys.stderr = logger

    return logger.log_dir


if __name__ == '__main__':
    init_log()
    print('Hello World !')
