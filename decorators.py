import time
import torch
import os
import random
from functools import wraps


def record_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('<' + func.__name__ + '>', str(end - start) + 's')
        return result

    return wrapper


class _save():
    def __init__(self, directory, filename, *, log=False, root='data'):
        self.directory = _save.toIterable(directory)
        self.filename = _save.toIterable(filename)
        self.root = _save.toIterable(root)
        if len(self.root) != 1 and len(self.root) != len(self.directory):
            raise ValueError('Length of root should be equal to length of directory.')
        if len(self.directory) != 1 and len(self.directory) != len(self.filename):
            raise ValueError('Length of directory should be equal to length of filename.')
        l = len(self.filename)
        if l != 1:
            if len(self.directory) == 1:
                self.directory = self.directory * l
            if len(self.root) == 1:
                self.root = self.root * l
        self.log = log

    @staticmethod
    def toIterable(obj: object):
        if not (isinstance(obj, tuple) or isinstance(obj, list)):
            obj = (obj,)
        return obj

    def __call__(self, func):
        @wraps(func)
        def with_saving(*args, **kwargs):
            result = func(*args, **kwargs)
            objects = _save.toIterable(result)
            if len(objects) != len(self.filename):
                raise ValueError('Redundant filenames')
            for obj, root, directory, filename in zip(objects, self.root, self.directory, self.filename):
                savedir = os.path.join(root, directory)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                savepath = os.path.join(root, directory, filename)
                self.save(obj, savepath)
                if self.log:
                    print('<' + func.__name__ + '>', 'saved at', savepath)
            return result

        return with_saving

    def get_savepath(self):
        paths = []
        for root, directory, filename in zip(self.root, self.directory, self.filename):
            paths.append(os.path.join(root, directory, filename))
        if len(paths) == 1:
            return paths[0]
        return paths

    def save(self, result, savepath):
        raise NotImplementedError


class torch_save(_save):
    def save(self, result, savepath):
        torch.save(result, savepath)


def probable(p=0.5, *, fail=None):
    """
    依照概率p执行函数
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rand = random.random()
            return func(*args, **kwargs) if rand <= p else fail

        return wrapper

    return decorator
