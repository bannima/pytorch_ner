#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version: 0.1
@author: zhouenguo
@license: Apache Licence
@file: trainer.py
@time: 2022/2/11 11:14 AM
@desc:
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from ner.modules.ner_loss import create_loss
from ner.modules.eval_metrics import create_metrics
from ner.config import logger
from ner.modules.utils import format_time, current_time, save_to_json
from ner.modules.utils import flatten

class Trainer():
    '''
    unified train procedure implementation, for single task
    '''

    def __init__(self,
                 model,
                 dataloaders,
                 result_path,
                 hypers,
                 raw_to_vector=None,
                 vector_to_raw=None,
                 **kwargs):
        super(Trainer).__init__()

        assert model is not None
        self.model = model

        self.criterion = create_loss(hypers.criterion)
        self.metrics = create_metrics(hypers.metrics)
        self.train_loader, self.valid_loader, self.test_loader = dataloaders
        self.raw_to_vector = raw_to_vector
        self.vector_to_raw = vector_to_raw

        if not os.path.exists(result_path):
            raise ValueError("result path not exists: {}".format(result_path))
        self.result_path = result_path

        self.hypers = hypers

        self._set_device()
        self._set_random_seed()

        # move model to GPU
        if torch.cuda.is_available():
            self.model.cuda()
            if self.n_gpu > 1:
                self.model = nn.DataParallel(self.model)

        # optimzer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.hypers.lr,
            eps=1e-8
        )

        # report the whole trainer
        self.summary()

    def _set_random_seed(self):
        # set the seed value all over the place to make this reproducible
        self.seed_val = 9527
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

    def _set_device(self):
        # set runtime env. GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
            logger.info("Using GPU, {} devices available".format(self.n_gpu))
        else:
            logger.info("Using CPU ... ")
            self.device = torch.device('cpu')

    def fit(self,output_loss=False):
        '''
        run epochs for train-valid-test procedure, and report statistics
        :return:
        '''
        # list to store a number of statistics
        epoch_stats = []

        # total_steps
        total_steps = len(self.train_loader) * self.hypers.epoch

        # create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        total_t0 = time.time()

        for epoch in tqdm(range(1, self.hypers.epoch + 1),
                          desc="Training All {} Epochs".format(self.hypers.epoch+ 1), unit='epoch'):
            logger.info("# Training for epoch: {} ".format(epoch))

            t0 = time.time()

            # put the model into training mode
            self.model.train()

            epoch_train_loss = self.train(epoch,output_loss)
            logger.info(" # Train loss for epoch:{} is {} ".format(epoch, epoch_train_loss))

            # mesure how long this epoch took
            training_time = format_time(time.time() - t0)

            t0 = time.time()
            # eval mode
            self.model.eval()

            epoch_eval_loss, eval_metrics = self.valid(epoch,output_loss, save_preds=self.hypers.save_val_preds)

            logger.info("# Valid loss for epoch {} is {} ".format(epoch, epoch_eval_loss))
            for metric_name in eval_metrics:
                logger.info(
                    " # Valid {} score for epoch {} is {}".format(metric_name, epoch, eval_metrics[metric_name]))

            # measure how long the validation run took
            valid_time = format_time(time.time() - t0)

            epoch_test_loss, test_metrics = self.test(epoch,output_loss, save_preds=self.hypers.save_test_preds)
            logger.info("# Test loss for epoch {} is {} ".format(epoch, epoch_test_loss))

            #can be none
            if test_metrics:
                for metric_name in test_metrics:
                    logger.info(" # Test {} score for epoch {} is {}".format(metric_name, epoch, test_metrics[metric_name]))

            # record all statistics in this epoch
            epoch_stats.append(
                {
                    "Epoch": epoch,
                    "Train Loss": epoch_train_loss,
                    "Trian Time": training_time,
                    "Valid Loss": epoch_eval_loss,
                    "Valid Time": valid_time,
                    "Eval Metrics": eval_metrics,
                    "Test Loss": epoch_test_loss,
                    "Test Metrics": test_metrics,
                    "Test Time": format_time(time.time() - t0),
                    "Time": current_time(),
                    "Test Result Dir": self.exp_result_dir
                }
            )

            if self.hypers.save_model:
                self.save_model(epoch)

        self.epoch_stats_file = self.save_epoch_statistics(epoch_stats)
        logger.info(
            "# Training complete; Total Train Procedure took: {}".format(str(format_time(time.time() - total_t0))))
        logger.info(
            "# Epoch Statistics saved at {}".format(self.epoch_stats_file)
        )

    def tensor_to_device(self,tensors):
        if torch.cuda.is_available():
            if isinstance(tensors,torch.Tensor):
                return tensors.to(self.device)
            elif isinstance(tensors,list):
                return [tensor.to(self.device) for tensor in tensors]
            elif isinstance(tensors,dict):
                return {
                    key:val.to(self.device) for key,val in tensors.items()
                }
        return tensors

    def train(self, epoch, output_loss):
        ''' train process '''
        total_train_loss = 0
        num_batchs = 0

        for batch in tqdm(self.train_loader, desc=" Train for Epoch: {}".format(epoch), unit='batch',colour='green'):
            num_batchs += 1
            # clear any previously calculated gradients before performing a backward pass
            self.model.zero_grad()

            raw_input, raw_label = batch

            # convert the raw input and label to numerical format
            if self.raw_to_vector is not None:
                (inputs_tensors,seq_lengths),labels_tensors = self.raw_to_vector(raw_input, raw_label)

            # move numerical tensor to GPU
            inputs_tensors = self.tensor_to_device(inputs_tensors)
            labels_tensors = self.tensor_to_device(labels_tensors)

            if not output_loss:
                outputs = self.model(inputs_tensors, seq_lengths, labels_tensors)

                batch_loss = self.calc_loss(outputs, labels_tensors, seq_lengths)
            else:
                batch_loss,batch_preds = self.model(inputs_tensors, seq_lengths, labels_tensors)

            # perform a backward pass to calculate the gradients
            batch_loss.backward()

            total_train_loss += batch_loss.item()

            # normalization of the gradients to 1.0 to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters and take a step using the computed gradient
            self.optimizer.step()

            # updating the learning rate
            self.scheduler.step()

        return total_train_loss

    def _forward_pass_with_no_grad(self, epoch, output_loss, loader, metrics, save_preds=False, type='Val',is_inference=False):
        ''' forward pass with no gradients, for validation and test. '''
        epoch_loss = 0
        raw_inputs = []
        predict_label = []
        target_label = []
        group_ids = []
        eval_metrics = None
        for batch in tqdm(loader, desc='{} for epoch {}'.format(type, epoch), unit="batch"):

            if len(batch)==2:
                raw_input, raw_label = batch
            else:
                raw_input = batch; raw_label=None

            # # convert the raw input and label to numerical format
            # if self.raw_to_vector is not None:
            #     (inputs, seq_lengths), labels = self.raw_to_vector(raw_input, raw_label)

            # convert the raw input and label to numerical format
            if self.raw_to_vector is not None:
                (inputs_tensors, seq_lengths), labels_tensors = self.raw_to_vector(raw_input, raw_label)

            # move numerical tensor to GPU
            inputs_tensors = self.tensor_to_device(inputs_tensors)

            # no gradients
            with torch.no_grad():
                # forward pass

                if not output_loss:
                    outputs = self.model(inputs_tensors, seq_lengths, labels_tensors)
                    loss = self.calc_loss(outputs, labels_tensors, seq_lengths)

                    # transform output logits to final NER type predictions, truncate with seq lengths
                    y_pred, labels_tensors = self.transform_predicts(outputs, labels_tensors, seq_lengths)

                else:
                    loss, y_pred = self.model(inputs_tensors, seq_lengths, labels_tensors)
                    if isinstance(labels_tensors, torch.Tensor):
                        labels_tensors.to('cpu')

                if not is_inference:
                    epoch_loss += loss.item()
                    target_label += labels_tensors

                predict_label += y_pred
                raw_inputs += raw_input

        #inference has no target labels
        if not is_inference:
            eval_metrics = self.calc_metrics(predict_label, target_label, metrics, group_ids)

        #self.report_metrics(eval_metrics)

        if save_preds:
            # save predicts and true labels, convert to raw label if exists self.vector_to_raw
            if not is_inference:
                data = pd.DataFrame(
                    {
                        'text':raw_inputs,
                        'predict': self.vector_to_raw(predict_label) if self.vector_to_raw is not None else predict_label,
                        'labels': self.vector_to_raw(target_label) if self.vector_to_raw is not None else target_label
                    }
                )
            else:
                data = pd.DataFrame(
                    {
                        'text': raw_inputs,
                        'predict': self.vector_to_raw(
                            predict_label) if self.vector_to_raw is not None else predict_label
                    }
                )
            self.save_predictions(epoch, data, type)

        return epoch_loss, eval_metrics

    def calc_metrics(self, predict_labels, target_labels, metrics, group_ids=None):
        ''' calc metrics for single task, can be override '''

        #flatten nested preds and labels to list
        predict_labels,target_labels = list(flatten(predict_labels)),list(flatten(target_labels))
        eval_metrics = {}
        for metric_name, metric in metrics.items():
            eval_metrics[metric_name] = metric(target_labels, predict_labels)
        return eval_metrics

    def report_metrics(self, metrics_results):
        ''' report the eval and test metrics '''
        for metric in metrics_results:
            logger.info("{}: {}".format(metric, metrics_results[metric]))

    def transform_predicts(self, outputs, labels, seq_lengths):
        ''' transform the predicts logits to final prediction format, which depends on the task type.
        current for NER, can be override'''

        # move logits and labels to GPU
        logits = outputs.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=-1)

        # truncate the predict labels with seq lengths
        y_pred = [list(seq_pred[:seq_length]) for (seq_length, seq_pred) in zip(seq_lengths.tolist(), y_pred)]

        #labels can be None, for inference case
        if labels is not None:
            if isinstance(labels,torch.Tensor):
                labels.to('cpu')

            # truncate the true labels with seq lengths
            labels = [list(seq_label[:seq_length]) for (seq_length, seq_label) in zip(seq_lengths.tolist(), labels.tolist())]

            return y_pred,labels

        return y_pred,None

    def calc_loss(self, outputs, labels,seq_lengths):
        '''single task loss calculation, can be override'''
        # labels = labels.view(-1,1) #tranform [batch_size] to [batch_size X 1]
        # labels = labels.to(torch.float32)
        batch_loss = self.criterion(outputs, labels, seq_lengths)
        return batch_loss

    def valid(self, epoch, output_loss=False,save_preds=False):
        ''' valid process '''
        logger.info(" Eval epoch {}".format(epoch))
        epoch_val_loss, val_metrics = self._forward_pass_with_no_grad(epoch, output_loss,self.valid_loader, self.metrics,
                                                                      save_preds=save_preds, type='Val')
        return epoch_val_loss, val_metrics

    def test(self, epoch, output_loss=False,save_preds=False):
        ''' test process '''
        logger.info(" Test epoch {}".format(epoch))
        epoch_test_loss, test_metrics = self._forward_pass_with_no_grad(epoch, output_loss, self.test_loader, self.metrics,
                                                                        save_preds=save_preds, type='Test',
                                                                        is_inference=True)
        return epoch_test_loss, test_metrics

    @property
    def exp_result_dir(self):
        unique_dir = 'Exp_LR{}_Batch{}_Loss{}'.format( \
            self.hypers.lr, self.hypers.batch_size, self.hypers.criterion)
        exp_result_dir = os.path.join(self.result_path, unique_dir)
        if not os.path.exists(exp_result_dir):
            os.mkdir(exp_result_dir)
        return exp_result_dir

    def save_model(self, epoch):
        model_filename = "Model_Epoch{}_Time{}.m".format(epoch, str(current_time()))
        model_filepath = os.path.join(self.exp_result_dir, model_filename)
        torch.save(self.model, model_filepath)
        logger.info("Model {} saved at {}".format(model_filename, self.exp_result_dir))

    def save_predictions(self, epoch, data,type):
        prediction_path = os.path.join(self.exp_result_dir, 'predicts')
        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)
        pred_filename = "Model_Epoch{}_{}_Predictions.json".format(epoch,type)
        pred_filepath = os.path.join(prediction_path, pred_filename)

        save_to_json(data.to_dict(orient='records'), pred_filepath)
        logger.info("Prediction {} saved at {}".format(pred_filename, self.exp_result_dir))

    def save_epoch_statistics(self, stats):
        ''' save statistics for each epoch '''
        epoch_stats_file = 'Epoch_Statstics_Time{}.csv'.format(str(current_time()))
        stats = pd.DataFrame(stats)
        stats.to_csv(os.path.join(self.exp_result_dir, epoch_stats_file), sep=',', encoding='utf-8', index=False)
        return os.path.join(self.exp_result_dir, epoch_stats_file)

    def summary(self):
        ''' summary the model '''
        pass
