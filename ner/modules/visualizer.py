#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: visualizer.py
@time: 2021/12/21 1:43 PM
@desc: 
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from ner.modules.utils import current_time

# chinese font file
font_file = os.path.join(os.path.dirname(__file__), 'project_file/HuaWenHeiTi-1.ttf')
zh_font = matplotlib.font_manager.FontProperties(
    fname=font_file
)


def draw_single_lines_chart(xticks, yticks, label, title="", save_path='./figs/', inter=None, rotation=30, fontsize=8):
    ''' 单条折线图 '''
    plt.cla()
    colors = ['blue', 'darkblue', 'red', 'orange', 'yellow']
    plt.plot(range(len(xticks)), yticks, linewidth=1.2, color=colors[0], label=label)
    if inter is None:
        inter = int(len(xticks) / 6)
    plt.xticks(range(0, len(xticks), inter), xticks[0:len(xticks):inter], fontproperties=zh_font)
    plt.legend(prop=zh_font)
    plt.title(title, fontproperties=zh_font)
    plt.savefig(os.path.join(save_path, title), type='jpg', dpi=600)
    plt.show()


def draw_twin_lines_chart(title, x_axis, ax1_yticks, ax1_metrics, ax2_yticks, ax2_metrics, xlabel, ax1_ylabel,
                          ax2_ylabel, save_path='./figs/'):
    ''' twin lines chart '''
    plt.clf()
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(xlabel, fontsize='large')

    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 1.4)
    for y_tick, metric in zip(ax1_yticks, ax1_metrics):
        ax1.plot(x_axis, y_tick, marker='.', lw=1.5)
    ax1.legend(ax1_metrics, loc='upper left', fontsize=10)
    ax1.set_ylabel(ax1_ylabel, fontsize='large')

    ax2 = ax1.twinx()
    for y_tick, metric in zip(ax2_yticks, ax2_metrics):
        ax2.plot(x_axis, y_tick, marker='*', lw=1.5, linestyle='--')
    ax2.legend(ax2_metrics, loc='upper right', fontsize=10)
    ax2.set_ylabel(ax2_ylabel, fontsize='large')

    plt.grid(ls='--')
    plt.title(title, color='black', fontsize='medium')

    title = "{}_{}".format(title,str(current_time()))
    plt.savefig(os.path.join(save_path, title), type='jpg', dpi=600)
    plt.show()
