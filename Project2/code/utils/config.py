"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Interface for reading config file
"""

import glob


try:
    with open('config.json') as f:
        config = eval(f.read())
except SyntaxError:
    print('Uh oh... I could not parse the config file. Is it typed correctly? --- utils.config ')
except IOError:
    print('Uh oh... I could not find the config file. --- utils.config')


def get(attr, root=config):
    ''' Return value of specified configuration attribute. '''
    node = root
    for part in attr.split('.'):
        node = node[part]
    return node


def print_if_verbose(s):
    ''' '''
    if get('DATA.VERBOSE'):
        print(s)


def is_file_prefix(attr):
    ''' Determine whether or not the filename under field `attr` in config.json
        is a prefix of any actual file. For instance, this comes in handy when
        seeing whether or not we have a model checkpoint saved.
    '''
    return bool(glob.glob(get(attr) + '*'))


def test():
    ''' Ensure reading works '''
    assert(get('META.AUTHOR') == '445 Staff')
    assert(get('META.AUTHOR') != '445Staff')
    print('utils.config test passed!')


if __name__ == '__main__':
    test()
