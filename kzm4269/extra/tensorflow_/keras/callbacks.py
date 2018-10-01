import csv

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList

__all__ = [
    'CallbackBranch',
    'BatchCounter',
    'CSVLogger',
    'ModelCheckpoint',
    'TensorBoard',
]


class CallbackBranch(CallbackList):
    @staticmethod
    def _copy_logs(logs):
        if logs is None:
            return logs
        return dict(logs)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, self._copy_logs(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, self._copy_logs(logs))

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, self._copy_logs(logs))

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, self._copy_logs(logs))

    def on_train_begin(self, logs=None):
        super().on_train_begin(self._copy_logs(logs))

    def on_train_end(self, logs=None):
        super().on_train_end(self._copy_logs(logs))


class BatchCounter(tf.keras.callbacks.Callback):
    def __init__(self, batches_key='batches', samples_key='samples'):
        super().__init__()
        self._batches = 0
        self._samples = 0
        self._batches_key = batches_key
        self._samples_key = samples_key

    def _update_logs(self, logs):
        if logs is None:
            return
        logs.update({
            self._batches_key: self._batches,
            self._samples_key: self._samples,
        })

    def on_train_begin(self, logs=None):
        self._update_logs(logs)

    def on_train_end(self, logs=None):
        self._update_logs(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._update_logs(logs)

    def on_epoch_end(self, epoch, logs=None):
        self._update_logs(logs)

    def on_batch_begin(self, epoch, logs=None):
        self._update_logs(logs)

    def on_batch_end(self, batch, logs=None):
        self._batches += 1
        self._samples += logs['size']
        self._update_logs(logs)


class CSVLogger(tf.keras.callbacks.CSVLogger):
    def __init__(self, *args, include_batch_logs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_batch_logs = include_batch_logs
        self._batch_logs = []

    def _logs(self, logs):
        return {key: logs.get(key, np.nan) for key in self.keys}

    def on_epoch_end(self, epoch, logs=None):
        if self.writer is None:
            self.keys = sorted(
                set(logs.keys()) | set(self._batch_logs[0].keys()))

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=['epoch'] + self.keys,
                dialect=CustomDialect,
            )
            if self.append_header:
                self.writer.writeheader()

        for batch_logs in self._batch_logs:
            super().on_epoch_end(epoch, self._logs(batch_logs))
        self._batch_logs.clear()

        super().on_epoch_end(epoch, self._logs(logs))

    def on_batch_end(self, batch, logs=None):
        if self.include_batch_logs:
            self._batch_logs.append(logs)


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, period_mode='epoch', **kwargs):
        allowed_period_modes = ('epoch', 'batch')
        if period_mode not in allowed_period_modes:
            raise ValueError('`period_mode` must be in {}, not {!r}'.format(
                allowed_period_modes,
                period_mode,
            ))

        super().__init__(*args, **kwargs)
        self.period_mode = period_mode
        self.epoch = 0

    def _on_epoch_or_batch_end(self, logs=None):
        super().on_epoch_end(self.epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.period_mode == 'epoch':
            self._on_epoch_or_batch_end(logs)

    def on_batch_end(self, batch, logs=None):
        if self.period_mode == 'batch':
            self._on_epoch_or_batch_end(logs)


class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        self._train_only = kwargs.pop('train_only', False)
        self._val_only = kwargs.pop('val_only', False)
        if self._train_only and self._val_only:
            raise ValueError('cannot use both `train_only` and `val_only`')
        super().__init__(*args, **kwargs)

    @staticmethod
    def _losses_and_metrics(logs):
        for key, value in dict(logs).items():
            if key in ['batch', 'size']:
                continue
            else:
                yield key, value

                if key == 'acc' or key.endswith('_acc'):
                    yield key[len('_acc'):] + '_err', 1.0 - value

    def _train_logs(self, logs):
        logs = dict(logs)
        losses_and_metrics = {key for key, _ in self._losses_and_metrics(logs)}

        for key, value in logs.items():
            if key in losses_and_metrics:
                if not key.startswith('val_'):
                    yield key, value
            else:
                yield key, value

    def _val_logs(self, logs):
        logs = dict(logs)
        losses_and_metrics = {key for key, _ in self._losses_and_metrics(logs)}

        for key, value in logs.items():
            if key in losses_and_metrics:
                if key.startswith('val_'):
                    yield key[len('val_'):], value
            else:
                yield key, value

    def _logs(self, logs):
        if logs is None:
            return logs
        assert isinstance(logs, dict)

        result = dict(logs)

        if self._train_only:
            result = dict(self._train_logs(result))
        if self._val_only:
            result = dict(self._val_logs(result))

        with np.errstate(divide='ignore'):
            result.update({
                key + '/log10': np.log10(value)
                for key, value in self._losses_and_metrics(result)
                if not (key == 'acc' or key.endswith('_acc'))
            })

        return result

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs=self._logs(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=self._logs(logs))
