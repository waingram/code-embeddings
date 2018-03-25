# -*- coding: utf-8 -*-

"""
code_embeddings.utils
This module provides utility functions that are used within code_embeddings.
"""
import os
import regex
from shutil import copyfile
from string import punctuation

source_dir = '/Users/waingram/Projects/RosettaCodeData'
dest_dir = '../data'

first_cap_re = regex.compile('(.)([A-Z][a-z]+)')
all_cap_re = regex.compile('([a-z0-9])([A-Z])')


def convert_camel_case(s):
    s1 = first_cap_re.sub(r'\1 \2', s)
    return all_cap_re.sub(r'\1 \2', s1).lower()


def convert_snake_case(s):
    return regex.sub(r"_", " ", s)


def tokenize(string):
    text = convert_snake_case(string)
    translator = str.maketrans(punctuation, ' ' * len(punctuation))  # map punctuation to space
    clean = text.translate(translator).split()
    # clean = ''.join([' ' if c in punctuation else c for c in text]).split()
    tokens = [e for e in (' '.join(clean)).lower().split()]
    return tokens


def _copy_files(task, lang):
    os.makedirs(os.path.join(dest_dir, lang, task), exist_ok=True)
    implementations = os.listdir(os.path.join(source_dir, 'Task', task, lang))
    for implementation in implementations:
        print(os.path.join(source_dir, 'Task', task, lang, implementation))
        copyfile(os.path.join(source_dir, 'Task', task, lang, implementation),
                 os.path.join(dest_dir, lang, task, implementation))


def create_dataset():
    tasks = [d for d in os.listdir(source_dir + '/Task') if
             os.path.exists(os.path.join(source_dir, 'Task', d, 'JavaScript')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Java')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Python')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'PHP')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Ruby')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'C++'))]
    for task in tasks:
        _copy_files(task, 'JavaScript')
        _copy_files(task, 'Java')
        _copy_files(task, 'Python')
        _copy_files(task, 'PHP')
        _copy_files(task, 'Ruby')
        _copy_files(task, 'C++')


def parse_subroutines_javascript(code):
    """Parse JavaScript files into separate functions

        :param code: JS code to parse.
        :rtype: map
    """
    pattern = r'function\s+\w+\s*\([^{]+({(?:[^{}]+\/\*.*?\*\/|[^{}]+\/\/.*?$|[^{}]+|(?1))*+})'
    scanner = regex.finditer(pattern, code, regex.MULTILINE)
    return map(lambda match: match.group(0), scanner)


def parse_subroutines_java(code):
    """Parse Java files into separate methods

        :param code: Java code to parse.
        :rtype: map
    """
    pattern = r'(?:(?:public|private|static|protected)\s+)*\s*[\w\<\>\[\]]+\s+\w+\s*\([^{]+({(?:[^{}]+\/\*.*?\*\/|[^{}]+\/\/.*?$|[^{}]+|(?1))*+})'
    scanner = regex.finditer(pattern, code, regex.MULTILINE)
    return map(lambda match: match.group(0), scanner)


def parse_subroutines_php(code):
    """Parse PHP files into separate functions

        :param code: PHP code to parse.
        :rtype: map
    """
    pattern = r'function\s+\w+\s*\([^{]+({(?:[^{}]+\/\*.*?\*\/|[^{}]+\/\/.*?$|[^{}]+|(?1))*+})'
    scanner = regex.finditer(pattern, code, regex.MULTILINE)
    return map(lambda match: match.group(0), scanner)
