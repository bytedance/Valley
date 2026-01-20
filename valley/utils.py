import datetime
import logging
import logging.handlers
import os
import sys
import requests

from valley.constants import LOGDIR
import re
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None
import logging
import sys

import torch.distributed as dist
from prettytable import PrettyTable



def print_trainable_params(model):
    if dist.get_rank() == 0:
        trainable_params = [k for k,v in model.named_parameters() if v.requires_grad]
        trainable_params_group = {}
        for para in trainable_params:
            layer_num = re.findall(r'layers.(\d+)\.',para)
            if layer_num:
                cur_layer = int(layer_num[0])
                if para.replace('layers.'+layer_num[0],'layers.*') not in trainable_params_group:
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
                elif cur_layer > int(trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')]):
                    trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
                    
            else:
                trainable_params_group[para] = '0'
        table = PrettyTable(['Parameter Name','Max Layer Number'])
        for key in trainable_params_group.keys():
            table.add_row([key, str(int(trainable_params_group[key])+1)])
        
        print(table)
        total_num = sum([v.numel() for k,v in model.named_parameters()])
        trainable_num = sum([v.numel() for k,v in model.named_parameters() if v.requires_grad])
        print('Total: {:.2f}M'.format(total_num/1e6), ' Trainable: {:.2f}M'.format(trainable_num/1e6))

def rank_zero_info(content: str, logger, print_type: str = "info"):
    output_method = getattr(logger, print_type)
    if dist.get_rank() == 0:
        output_method(content)


def get_logger(name: str):
    # logger initialize
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    # add handler
    logger.addHandler(handler)

    return logger


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
