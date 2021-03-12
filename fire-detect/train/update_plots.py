import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn
import matplotlib.pyplot as plt
import sys
import glob
from global_config import *
import threading
import time

from ev_toolkit import plot_tool


def save_plot(event_file, tags):
    """从event_file读取Tensorboard数据，并将画图数据更新到远程界面
    """
    if not os.path.isfile(event_file):
        print(f'{event_file} 不存在')
        return
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 1,
        'scalars': 100,
        'histograms': 1
    }
    event_acc = EventAccumulator(event_file, tf_size_guidance)
    event_acc.Reload()
    for tag in tags:
        try:
            event = event_acc.Scalars(tag)
        except KeyError:
            print(f'Tag {tag} does not exist!')
            continue
        steps = []
        data = []
        for event in event:
            steps.append(event[1])
            data.append(event[2])
        seaborn.set()
        plt.clf()
        plt.rcParams['figure.figsize'] = (16, 8)
        plt.plot(steps, data)
        plt.xlabel('steps')
        plt.ylabel(f'{tag}')
        plt.title(f"update time:{time.time()}")
        plot_name = f"{tag.replace('/', '_')}"
        print(f'Updating plot with name:{plot_name}')
        plot_tool.update_plot(plot_name=plot_name, img=plt.gcf())

def plot_timer():
    while True:
        tag_list = ['LearningRate/LearningRate/learning_rate', 'Losses/TotalLoss', 'Losses/Loss/localization_loss',
                'Losses/Loss/classification_loss']
        event_file_list = glob.glob(f'{project_root}/training/events.out.tfevents.*')
        if len(event_file_list) == 0:
            print('Tensorboard event file not found!')
        else:
            save_plot(event_file_list[0], tag_list)
        time.sleep(5)

if __name__ == '__main__':
    t = threading.Thread(target=plot_timer)
    t.start()
