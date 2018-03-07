# build dataset from RosettaData project
import os

base_dir = '/Users/waingram/Projects/RosettaCodeData'

if __name__ == '__main__':
    tasks = [d for d in os.listdir(base_dir + '/Task') if os.path.exists(os.path.join(base_dir, 'Task', d, 'JavaScript')) and
                                                          os.path.exists(os.path.join(base_dir, 'Task', d, 'Java')) and
                                                          os.path.exists(os.path.join(base_dir, 'Task', d, 'Python')) and
                                                          os.path.exists(os.path.join(base_dir, 'Task', d, 'PHP')) and
                                                          os.path.exists(os.path.join(base_dir, 'Task', d, 'Ruby')) and
                                                          os.path.exists(os.path.join(base_dir, 'Task', d, 'C++'))]
    for task in tasks:
        print(os.path.join(base_dir, 'Task', task))
