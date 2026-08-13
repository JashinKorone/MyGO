# coding: utf-8

r"""Training loop.

Two paper-specific behaviours are implemented here:

* ``post_epoch_processing`` is called after every epoch so that the PMA prompt
  global memory can close the current epoch zone (Fig. 5);
* evaluation additionally reports the per modality-combination accuracy / F1 of
  Table 5, which is the standard protocol for modality-incomplete detection.
"""

import torch
import torch.optim as optim
from logging import getLogger
from time import time

from utils.evaluator import combination_metrics, metrics, prompt_to_modalities
from utils.utils import dict2str, early_stopping


class AbstractTrainer(object):
    r"""Manage the training and evaluation processes."""

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [fit] should be implemented.')

    def evaluate(self, eval_data):
        raise NotImplementedError('Method [evaluate] should be implemented.')


class Trainer(AbstractTrainer):
    r"""Basic trainer with early stopping and learning-rate scheduling."""

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['batch_size']
        self.device = config['device']
        self.weight_decay = config['weight_decay']
        self.clip_grad_norm = config['clip_grad_norm']
        self.report_combination = (
            config['report_combination'] if config['report_combination'] is not None else True
        )

        self.start_epoch = 0
        self.cur_step = 0

        self.metrics = config['metrics']
        tmp_dd = {f'{j.lower()}': 0.0 for j in self.metrics}
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.best_combination_result = {}
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        if learner == 'adam':
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'adamw':
            return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'sgd':
            return optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'adagrad':
            return optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'rmsprop':
            return optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.logger.warning('Received unrecognized optimizer, set default AdamW optimizer')
        return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _to_device(self, batch_data):
        for k, v in batch_data.items():
            batch_data[k] = v.to(self.device)
        return batch_data

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, batch_data in enumerate(train_data):
            batch_data = self._to_device(batch_data)
            self.optimizer.zero_grad()
            losses = loss_func(batch_data)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info(
                    'Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx)
                )
                return loss, torch.tensor(0.0)
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        result = self.evaluate(valid_data)
        valid_result = {f'{j.lower()}': result[f'{j.lower()}'] for j in self.metrics}
        return result[self.valid_metric], valid_result, result.get('combination', {})

    def _check_nan(self, loss):
        return bool(torch.isnan(loss))

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output += ', '.join(
                'train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses)
            )
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    @staticmethod
    def _format_combination(combination):
        lines = []
        for prompt in sorted(combination.keys()):
            item = combination[prompt]
            lines.append(
                '  {}  {:<18} n={:<5} acc={:.4f}  f1={:.4f}'.format(
                    prompt, prompt_to_modalities(prompt), item['num'], item['acc'], item['f1']
                )
            )
        return '\n'.join(lines)

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, val=True, verbose=True):
        r"""Train the model, returning the best validation / test results."""
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break  # nan loss
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            # Closes the current PMA memory zone and reports the weak prompts.
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            valid_start_time = time()
            eval_source = valid_data if val else test_data
            valid_score, valid_result, _ = self._valid_epoch(eval_source)
            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                valid_score, self.best_valid_score, self.cur_step,
                max_step=self.stopping_step, bigger=self.valid_metric_bigger,
            )
            valid_end_time = time()

            if val:
                _, test_result, test_combination = self._valid_epoch(test_data)
            else:
                test_result, test_combination = valid_result, {}

            if verbose:
                self.logger.info(
                    'epoch %d evaluating [time: %.2fs, valid_score: %f]'
                    % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                )
                self.logger.info('valid result: \n' + dict2str(valid_result))
                self.logger.info('test result: \n' + dict2str(test_result))

            if update_flag:
                if verbose:
                    self.logger.info(
                        '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    )
                self.best_valid_result = valid_result
                self.best_test_upon_valid = test_result
                self.best_combination_result = test_combination
                if self.report_combination and test_combination and verbose:
                    self.logger.info(
                        'modality-wise test result:\n' + self._format_combination(test_combination)
                    )
                if saved:
                    self.model.save_best()

            if stop_flag:
                if verbose:
                    self.logger.info(
                        '+++++Finished training, best eval result in epoch %d'
                        % (epoch_idx - self.cur_step * self.eval_step)
                    )
                break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model, including the per modality-combination results."""
        self.model.eval()

        tpred, tlabel, tprompt = [], [], []
        for batch_data in eval_data:
            batch_data = self._to_device(batch_data)
            preds = self.model.predict(batch_data)
            _, preds = torch.max(preds, 1)
            labels = batch_data['label']
            prompt = batch_data.get('prompt', batch_data.get('masker'))

            tlabel.extend(labels.detach().cpu().numpy().tolist())
            tpred.extend(preds.detach().cpu().numpy().tolist())
            if prompt is not None:
                tprompt.extend(prompt.detach().cpu().int().numpy().tolist())

        results = metrics(tlabel, tpred)
        if self.report_combination and tprompt:
            results['combination'] = combination_metrics(tlabel, tpred, tprompt)
        return results
