import numpy as np
import torch
import random


def add_to_writer(metrics: dict, writer, curr_step: int, sample_number=0):
    writer.add_scalar('Valid/Loss', metrics['loss'], curr_step)
    writer.add_scalar('Valid/Metric', metrics['metric'], curr_step)
    writer.add_text('Valid/Input_text', str(metrics['input_text'][sample_number]), curr_step)
    writer.add_text('Valid/Label_text', str(metrics['label'][sample_number]), curr_step)
    writer.add_text('Valid/Prediction_text', str(metrics['prediction'][sample_number]), curr_step)


def set_random_seed(random_seed: int):
    # set random seed for PyTorch and CUDA
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    # set random seed for Numpy
    np.random.seed(random_seed)

    # set random seed for random
    random.seed(random_seed)


def strList2intList(str_list: str) -> list:
    """
    :param str_list: e.g. "1, 2, 3"
    :return: [1, 2, 3]
    """
    str_list = str_list.split(',')
    int_list = [int(char) for char in str_list]
    return int_list


def str2bool(string: str) -> bool:
    """
    :param string: e.g. "True"
    :return: bool type
    """
    string = string.lower()
    if string == 'true':
        return True
    else:
        return False
