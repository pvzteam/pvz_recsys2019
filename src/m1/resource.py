# -*- coding: utf-8 -*-

import os
import sys
import datetime
import functools

from utils import load_dataframe, save_dataframe


DEBUG = False


class Resource(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.executor = None
        self.dependencies = set()

    def exists(self):
        return os.path.exists(self.path)

    def add_dependency(self, dependency):
        if isinstance(dependency, Resource):
            self.dependencies.add(dependency)
        elif isinstance(dependency, TrainTestResource):
            self.dependencies.add(dependency.train)
            self.dependencies.add(dependency.test)
        else:
            raise 'Invalid dependency, %s' % dependency

    def add_dependencies(self, dependencies):
        if isinstance(dependencies, list) or isinstance(dependencies, tuple):
            for dependency in dependencies:
                self.add_dependency(dependency)
        else:
            self.add_dependency(dependencies)

    def set_executor(self, func, *args, **kwargs):
        self.executor = (func, args, kwargs)

    def execute(self):
        if self.exists():
            return

        for dependency in self.dependencies:
            dependency.execute()

        assert self.executor, ('Resource not Registered, %s.'
                                % self.name)
        func, args, kwargs = self.executor
        
        if DEBUG:
            print('Start execute %s at %s.' % (func, datetime.datetime.now()))
        func(*args, **kwargs)
        

    def load(self, *args, **kwargs):
        raise Exception('Function not implemented.')

    def save(self, *args, **kwargs):
        raise Exception('Function not implemented.')

    def remove(self, *args, **kwargs):
        raise Exception('Function not implemented.')

    def __repr__(self):
        return 'Resource(%s)<%s>' % (self.name, self.path)


class DFResource(Resource):
    def __init__(self, folder, name, fmt):
        self.folder = folder

        assert fmt in ('ftr', 'csv', 'tsv')
        self.fmt = fmt

        if name.endswith(fmt):
            name = name[:-len(fmt)]

        path = os.path.join(folder, '%s.%s' % (name, fmt))
        super(DFResource, self).__init__(name, path)

    def load(self, *args, **kwargs):
        self.execute()

        dct = {'fmt': self.fmt}
        dct.update(kwargs)
        
        return load_dataframe(self.path, *args, **dct)

    def save(self, df, *args, **kwargs):
        # if self.exists():
        #     raise Exception('Resource already exists, %s.' % self.name)

        return save_dataframe(df, self.path, self.fmt, *args, **kwargs)

    def remove(self):
        if self.exists():
            os.remove(self.path)


class InputResource(DFResource):
    def __init__(self, name, fmt='csv', read_only=True):
        super(InputResource, self).__init__('../../input/', name, fmt)
        self.read_only = read_only

    def save(self, df, *args, **kwargs):
        if self.read_only:
            raise Exception('Input Resource is read only, %s' % self.name)

        return super(InputResource, self).save(df, *args, **kwargs)
        

class TmpResource(DFResource):
    def __init__(self, name, fmt='ftr'):
        super(TmpResource, self).__init__('../../tmp/', name, fmt)


class FeatResource(DFResource):
    def __init__(self, name, fmt='ftr'):
        super(FeatResource, self).__init__('../../feat/', name, fmt)


class ModelResource(DFResource):
    def __init__(self, name, fmt='csv'):
        super(ModelResource, self).__init__('../../model/', name, fmt)


class TrainTestResource(object):
    def __init__(self, cls, name_tp, fix=['train', 'test'], 
                 fmt='ftr', **kwargs):
        assert cls in (InputResource, TmpResource, 
                       FeatResource, ModelResource)

        self.cls = cls 
        self.name_tp = name_tp
        self.fix = fix
        self.train = cls(name_tp % fix[0], fmt=fmt, **kwargs)
        self.test = cls(name_tp % fix[1], fmt=fmt, **kwargs)

    @property
    def dependencies(self):
        return self.train.dependencies | self.test.dependencies

    def add_dependency(self, dependency):
        self.train.add_dependency(dependency)
        self.test.add_dependency(dependency)

    def add_dependencies(self, dependenies):
        self.train.add_dependencies(dependenies)
        self.test.add_dependencies(dependenies)

    def __getitem__(self, key):
        assert key in ('train', 'test')
        if key == 'train':
            return self.train 
        else:
            return self.test

    def exists(self):
        return self.train.exists() and self.test.exists()

    def set_executor(self, func, *args, **kwargs):
        self.train.set_executor(func, *args, **kwargs)
        self.test.set_executor(func, *args, **kwargs)

    def execute(self):
        self.train.execute()
        self.test.execute()

    def load(self, *args, **kwargs):
        raise Exception('Function not implemented.')

    def save(self, *args, **kwargs):
        raise Exception('Function not implemented.')

    def __repr__(self):
        return '%s(%s)<%s>' % (self.cls, self.name_tp, 
                               '|'.join(self.fix))


def _prepare_resource_list(rl):
    if rl is None:
        rl = []
    elif not (isinstance(rl, list) or isinstance(rl, tuple)):
        rl = [rl]
    else:
        rl = list(rl)

    for i in rl:
        assert (isinstance(i, Resource) or 
                isinstance(i, TrainTestResource)), (
                    'Invalid resource %s.' % i)

    return rl


def register(out=[], inp=[], *args, **kwargs):
    assert out, 'Empty output'

    out = _prepare_resource_list(out)
    inp = _prepare_resource_list(inp)

    def wrap(func):
        @functools.wraps(func)
        def inner_wrap(*fargs, **fkwargs):
            return func(*fargs, **fkwargs)
            
        for oi in out:
            oi.set_executor(func, *args, **kwargs)
            oi.add_dependencies(inp)

        return inner_wrap

    return wrap


def execute(*rs):
    for r in rs:
        r.execute()



