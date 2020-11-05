from itertools import product
from copy import deepcopy
from functools import reduce

class SimpleDictIterator(object):
    def __init__(self, d, fixed_keys=[]):
        iter_dict = {}
        for k, v in d.items():
            if isinstance(v, list) and k not in fixed_keys:
                iter_dict[k] = v
        self.iter_vals = product(*iter_dict.values())
        self.iter_dict = iter_dict
        self.fixed_keys = fixed_keys
        self.dict = d

    def __iter__(self):
        return self

    def __next__(self):
        d = deepcopy(self.dict)
        vals = self.iter_vals.__next__()
        for k, v in zip(self.iter_dict.keys(), vals):
            d[k] = v
        return d

# immutable wrapper class of tuple. Required for hashing of lists into dictionary, and distinguish from tuple
class KeyChain(tuple):
    def __init__(self, l):
        super(KeyChain, self).__init__()

class NestedDict(object):
    # keys consist of a list of keys. Keys are either single values or lists of keys for nested dictionaries
    # or tuples that represent groupings of the former two kinds.
    # We assume a maximum depth of a tuple of lists; beyond this depth groupings become ambiguous.
    # No validation is performed on key values.
    def __init__(self, keys, values, dot='__', comma=','):
        tmp = []
        # process all keys into hashable KeyChain instances. Tuples of keys will be processed into tuples of KeyChains
        for key in keys:
            if isinstance(key, tuple):
                key = tuple([KeyChain(k) if isinstance(k, list) else KeyChain([k])
                             for k in key])
                tmp.append(key)
            elif isinstance(key, list):
                tmp.append(KeyChain(key))
            else:
                tmp.append(KeyChain([key]))
        keys = tmp
        flat_dict = dict(zip(keys, values))
        # now must unpack tuples from keys and values (expand by width)
        tmp_keys, tmp_values = [], []
        for key, value in zip(keys, values):
            # if key is not a KeyChain, it must be a tuple of multiple keys. Proceed to unpack.
            if not isinstance(key, KeyChain):
                assert isinstance(value, tuple)
                tmp_keys.extend(key)
                tmp_values.extend(value)
            else:
                tmp_keys.append(key)
                tmp_values.append(value)
        keys = tmp_keys
        values = tmp_values
        # now construct nested dict (expand by depth)
        nested_dict = {}
        for key_chain, value in zip(keys, values):
            key_chain = list(key_chain)
            key_pointer, key_leaf = key_chain[:-1], key_chain[-1]
            pointer = reduce(lambda d, k: d.setdefault(k, {}), [nested_dict] + key_pointer)
            pointer[key_leaf] = value

        self.flat_dict = flat_dict
        self.dot = dot
        self.comma = comma
        self.dict = nested_dict
    
    # pass a dict with string keys. Tuple groupings are marked by a comma ','
    # and nested keys are marked by '___'.
    # grouped keys must have grouped values in a tuple.
    @classmethod
    def fromdict(cls, d, dot='__', comma=','):
        # order of operations: first split by comma, then by 'dot' operator.
        keys = []
        for key in d.keys():
            tmp = key.split(comma)
            tmp = [k.split(dot) for k in tmp]
            if len(tmp) == 1:
                keys.append(tmp[0])
            else:
                if tmp[-1] == '':  # if key ends in comma, remove last empty entry (for singleton tuples)
                    tmp = tmp[:-1]
                keys.append(tuple(tmp))
        values = d.values()
        return cls(keys, values, dot=dot, comma=comma)

    def __getitem__(self, key):
        if isinstance(key, str):
            if self.comma in key:
                key = tuple(key.split(self.comma))
                if key[-1] == '':
                    key = key[:-1]
            else:
                key = key.split(self.dot)
        elif not isinstance(key, list):
            key = [key]

        if isinstance(key, list):
            return reduce(lambda d, k: d[k], [self.dict] + key)
        else:
            assert isinstance(key, tuple)
            return tuple(self.dict[k] for k in key)


    def __setitem__(self, key, value):
        if isinstance(key, str):
            if self.comma in key:
                key = tuple(key.split(self.comma))
                if key[-1] == '':
                    key = key[:-1]
            else:
                key = key.split(self.dot)
        elif not isinstance(key, list):
            key = [key]

        if isinstance(key, list):
            key_pointer, key_leaf = key[:-1], key[-1]
            pointer = reduce(lambda d, k: d.setdefault(k, {}), [self.dict] + key_pointer)
            pointer[key_leaf] = value
            self.flat_dict[KeyChain(key)] = value
        else:
            assert isinstance(key, tuple) and isinstance(value, tuple)
            for k, v in zip(key, value):
                self.__setitem__(k, v)
    
    def asdict(self):
        return self.dict
    
    # the keys, values and items return respective objects with groupings preserved.
    # This is meant to facilitate grouped iteration by IterDict.
    def keys(self, tostring=False):
        if tostring:
            def process_key(key):
                if not isinstance(key, KeyChain):  # if key is a tuple, must have doubly nested loop of processing
                    if len(key) == 1:
                        return str(key[0]) + self.comma
                    return self.comma.join(self.dot.join(str(k) for k in key_) for key_ in key)
                else:
                    return self.dot.join(str(k) for k in key)
        
            return map(process_key, self.flat_dict.keys())
        else:
            return (list(key) if isinstance(key, KeyChain)
                    else tuple(list(k) for k in key)
                    for key in self.flat_dict.keys())
    
    def values(self):
        return self.flat_dict.values()

    def items(self, **kwargs):
        return zip(self.keys(**kwargs), self.values())

    def __repr__(self):
       return '%s(dict=%s, flat_dict=%s)' % (self.__class__.__name__, self.dict, self.flat_dict)

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))
        return new

class DictIterator(object):
    def __init__(self, d, fixed_keys=[], dot='__', comma=','):
        # locate keys that require iteration
        # Note that if a fixed_key is passed as part of a grouped key, it will be included in iteration.
        # Since the values must be a list of tuples, this is intended behavior that requires explicit iteration.
        iter_dict = {}
        for k, v in d.items():
            if isinstance(v, list) and k not in fixed_keys:
                iter_dict[k] = v

        # initialize nested dict structure. Due to unpacking check, initialize with first item of each list in iter_dict
        #        iter_dict = NestedDict.fromdict(iter_dict, dot=dot, comma=comma)
        self.iter_vals = product(*iter_dict.values())
        self.iter_dict = iter_dict
        self.fixed_keys = fixed_keys
        d_ = deepcopy(d)
        for key, vals in iter_dict.items():
            d_[key] = vals[0]
        self.dict = NestedDict.fromdict(d_, dot=dot, comma=comma)

    def __iter__(self):
        return self

    def __next__(self):
        d = deepcopy(self.dict)
        vals = self.iter_vals.__next__()
        for k, v in zip(self.iter_dict.keys(), vals):
            d[k] = v
        return d.asdict()

    def iterkeys(self):
        return self.iter_dict.keys()

if __name__ == '__main__':
    d = {'a': [1,2,3], 'b': [True, False], 'c': ['foo', 'bar']}
    for d_ in SimpleDictIterator(d, fixed_keys=['c']):
        print(d_)
    d = dict(a='A', b='B', x__y__z=[1,2], x__y__w = 'baz', d__e__f__g='spam')
    d['c,d'] = ('cal', 'day')
    nd = NestedDict.fromdict(d)
    print(nd)
    nd[['new','entry']] = 'changed'
    nd['d,e'] = ('don', 'erst')
    nd['singleton,'] = (111,)
    print(nd)
    for k, v in nd.items():
        print(k, '\t', v)
    print(nd)
    for k, v in nd.items(tostring=True):
        print(k, '\t', v)
    d = dict(a=1, b=[3,4], c__x=[1,2,3], c__y=['spam', 'eggs'], c__z__t=['foo','bar'])
    d = {'c__x': [1,2], 'b': ['bar', 'baz']}
    d['d,e']= [('re', 'in'), (2, 4)]
    d['f,'] = [('unfixed',), ('free',)]
    for d_ in DictIterator(d, fixed_keys=['c__x', 'f']):
        print(d_)
    
    d = dict(a=1, b=[3,4], c__x=[1,2,3], c__y=['spam', 'eggs'], c__z__t=['foo','bar'])
    d = {'c__x': [1,2], 'b': ['bar', 'baz']}
    d['d,e']= [('re', 'in'), (2, 4)]
    d['f'] = ['fixed', 'list']
    for d_ in DictIterator(d, fixed_keys=['c__x', 'f']):
        print(d_)
