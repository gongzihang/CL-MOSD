import sys
import datetime
import logging


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


from torch.utils.tensorboard import SummaryWriter
from accelerate.tracking import GeneralTracker, on_main_process
import os
from typing import Union


# 0. 自定义追踪器
class MyCustomTracker(GeneralTracker):
    """
    source:https://blog.csdn.net/EstrangedZ/article/details/132880245
    my custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike],
                 **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)
        
    def add_image(self, tag, image, **kwargs):
        self.writer.add_image(tag=tag, img_tensor=image, **kwargs)
                        