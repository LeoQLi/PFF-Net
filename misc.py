import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
import subprocess
from datetime import datetime


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # if seed == 0:  # slower, more reproducible
    #     torch.backends.cudnn.benchmark = False    # default is False
    #     torch.backends.cudnn.deterministic = True
    # else:          # faster, less reproducible
    #     torch.backends.cudnn.benchmark = True    # if True, the net graph and input size should be fixed !!!
    #     torch.backends.cudnn.deterministic = False


def git_commit(logger, log_dir=None, git_name=None):
    """
        Logs source code configuration
    """
    import git

    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        # if log_dir is not None:
        #     with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
        #         subprocess.run(['git', 'diff'], stdout=fid)

        git_name = git_name if git_name is not None else datetime.now().strftime("%y%m%d_%H%M%S")
        os.system("git add --all")
        os.system("git commit --all -m '{}'".format(git_name))
    except git.exc.InvalidGitRepositoryError:
        pass


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Output and logs will be saved to: {}'.format(log_dir))
    return logger


def get_new_log_dir(root='./logs', prefix='', postfix=''):
    name = prefix + time.strftime("%y%m%d_%H%M%S", time.localtime()) + postfix
    log_dir = os.path.join(root, name)
    os.makedirs(log_dir)
    return log_dir, name


def prepare(args):
    PID = os.getpid()

    log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                            postfix='_' + args.tag if args.tag is not None else '')
    sub_log_dir = os.path.join(log_path, 'log')
    os.makedirs(sub_log_dir)
    logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
    git_commit(logger=logger, log_dir=sub_log_dir, git_name=log_dir_name)

    ckpt_dir = os.path.join(log_path, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    code_dir = os.path.join(log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('net', code_dir))

    return logger, ckpt_dir