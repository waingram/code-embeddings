# build dataset from RosettaData project
import os
from shutil import copyfile

source_dir = '/Users/waingram/Projects/RosettaCodeData'
dest_dir = '../test_data'


def copy_files(task, lang):
    os.makedirs(os.path.join(dest_dir, lang, task), exist_ok=True)
    implementations = os.listdir(os.path.join(source_dir, 'Task', task, lang))
    for implementation in implementations:
        print(os.path.join(source_dir, 'Task', task, lang, implementation))
        copyfile(os.path.join(source_dir, 'Task', task, lang, implementation),
                 os.path.join(dest_dir, lang, task, implementation))


if __name__ == '__main__':
    tasks = [d for d in os.listdir(source_dir + '/Task') if
             os.path.exists(os.path.join(source_dir, 'Task', d, 'JavaScript')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Java')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Python')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'PHP')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'Ruby')) and
             os.path.exists(os.path.join(source_dir, 'Task', d, 'C++'))]
    for task in tasks:
        copy_files(task, 'JavaScript')
        copy_files(task, 'Java')
        copy_files(task, 'Python')
        copy_files(task, 'PHP')
        copy_files(task, 'Ruby')
        copy_files(task, 'C++')
