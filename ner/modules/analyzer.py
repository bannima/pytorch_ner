#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: analyzer.py
@time: 2021/12/25 3:59 PM
@desc: 
"""
from abc import ABCMeta, abstractmethod
import pandas as pd
from ner.modules.visualizer import draw_twin_lines_chart

class ExperimentAnalyzer(metaclass=ABCMeta):
    ''' Analysis the whole experiment '''

    def __init__(self, stats_file):
        self.statistics = pd.read_csv(stats_file, sep=',', encoding='utf-8')

    @abstractmethod
    def analysis_experiment(self, exp_result_dir, title):
        raise NotImplementedError


class SingleTaskExpAnalyzer(ExperimentAnalyzer):
    ''' Analysis the Single task Experiment '''

    def __init__(self, stats_file):
        super(SingleTaskExpAnalyzer, self).__init__(stats_file)

    def analysis_task_metric(self, epoch_metrics):
        eval_metrics = {}
        for epoch_metric in epoch_metrics:
                for metric in epoch_metric:
                    if metric not in eval_metrics:
                        eval_metrics[metric] = []
                    eval_metrics[metric].append(epoch_metric[metric])
        return list(eval_metrics.values()), list(eval_metrics.keys())

    def analysis_experiment(self, exp_result_dir, title):
        # load experiment statistics
        evals, eval_metric_names = self.analysis_task_metric(self.statistics['Eval Metrics'].apply(eval))

        loss = (self.statistics['Train Loss'], self.statistics['Valid Loss'])
        loss_metric_names = ("Train Loss", 'Valid Loss', 'Test Loss')

        draw_twin_lines_chart(title=title, \
                              x_axis=self.statistics['Epoch'], \
                              ax1_yticks=evals, \
                              ax1_metrics=eval_metric_names, \
                              ax2_yticks=loss, \
                              ax2_metrics=loss_metric_names, \
                              xlabel='Epochs', \
                              ax1_ylabel='Eval Metric', \
                              ax2_ylabel='Loss', \
                              save_path=exp_result_dir)


class MultiTaskExpAnalyzer(ExperimentAnalyzer):
    ''' Analysis the Multi task Experiment '''

    def __init__(self, stats_file):
        super(MultiTaskExpAnalyzer, self).__init__(stats_file)

    def analysis_task_metric(self, epoch_metrics):
        task_metrics = {}
        for epoch_metric in epoch_metrics:
            # epoch_metric = json.loads(epoch_metric)
            for task in epoch_metric:
                for metric in epoch_metric[task]:
                    key = "Task{}_{}".format(task, metric)
                    if key not in task_metrics:
                        task_metrics[key] = []
                    task_metrics[key].append(epoch_metric[task][metric])
        return list(task_metrics.values()), list(task_metrics.keys())

    def analysis_experiment(self, exp_result_dir, title):
        # load experiment statistics
        evals, eval_metric_names = self.analysis_task_metric(self.statistics['Test Metrics'].apply(eval))

        loss = (self.statistics['Train Loss'], self.statistics['Valid Loss'], self.statistics['Test Loss'])
        loss_metric_names = ("Train Loss", 'Valid Loss', 'Test Loss')

        draw_twin_lines_chart(title=title, \
                              x_axis=self.statistics['Epoch'], \
                              ax1_yticks=evals, \
                              ax1_metrics=eval_metric_names, \
                              ax2_yticks=loss, \
                              ax2_metrics=loss_metric_names, \
                              xlabel='Epochs', \
                              ax1_ylabel='Eval Metric', \
                              ax2_ylabel='Loss', \
                              save_path=exp_result_dir)
