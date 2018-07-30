import csv
import re
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CallbackList


class BatchCounter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch = -1
        self.nbatches = 0
        self.nsamples = 0

    def _update_logs(self, logs):
        if logs is None:
            return
        logs.update(
            batch=self.batch,
            nbatches=self.nbatches,
            nsamples=self.nsamples)

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
        self.batch = batch
        self.nbatches += 1
        self.nsamples += logs['size']
        self._update_logs(logs)


class CallbackBranch(CallbackList):
    def __init__(self, callbacks=None, queue_length=10, **kwargs):
        super().__init__(callbacks=callbacks, queue_length=queue_length,
                         **kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        if logs is not None:
            logs = dict(logs)
        super().on_train_end(logs)


class CSVLogger(tf.keras.callbacks.CSVLogger):
    def __init__(self, filename, separator=',', append=False,
                 include_batch_logs=False):
        super().__init__(filename=filename, separator=separator, append=append)
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
                dialect=CustomDialect)
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
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto',
                 period=1, period_mode='epoch'):
        allowed_period_modes = ('epoch', 'batch')
        if period_mode not in allowed_period_modes:
            raise ValueError('`period_mode` must be in {}, not {!r}'.format(
                allowed_period_modes, period_mode))

        super().__init__(
            filepath, monitor=monitor, verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only, mode=mode, period=period)
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
    def __init__(self, log_dir, *args, **kwargs):
        super().__init__(log_dir + '/val', *args, **kwargs)
        self.train_writer = tf.summary.FileWriter(log_dir + '/tr')

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch)

        train_summary = tf.Summary()
        val_summary = tf.Summary()

        groups = defaultdict(list)
        for name, value in logs.items():
            groups[re.sub(r'_\d+([_/]|$)', r'_*\1', name)].append(value)
        logs = dict(logs, **{k: np.mean(v, axis=0) for k, v in groups.items()})

        for name, value in logs.items():
            if name in {'size', 'batch'}:
                continue
            summary = val_summary if name.startswith('val_') else train_summary
            name = name[4:] if name.startswith('val_') else name
            summary.value.add(
                tag=name,
                simple_value=value)
            if name.split('_')[-1] == 'acc':
                summary.value.add(
                    tag=re.sub(r'acc$', 'err', name),
                    simple_value=1 - value)
                summary.value.add(
                    tag=re.sub(r'acc$', 'err', name) + '/log10',
                    simple_value=np.log10(1 - value))
            else:
                summary.value.add(
                    tag=name + '/log10',
                    simple_value=np.log10(value))

        self.train_writer.add_summary(train_summary, epoch)
        self.writer.add_summary(val_summary, epoch)

        self.train_writer.flush()
        self.writer.flush()


def _main():
    import tempfile

    n = 100
    y = x = tf.keras.layers.Input((n,))
    for _ in range(3):
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.Dense(n, activation='softsign')(y)
    model = tf.keras.models.Model([x], [y])
    model.compile('adam', 'mse')
    model.summary()

    with tempfile.TemporaryDirectory() as tempd:
        train_x = np.random.random([10000, n])
        train_y = np.sin(train_x)
        try:
            model.fit(
                train_x, train_y,
                batch_size=100, epochs=10, verbose=2, validation_split=0.1,
                callbacks=[
                    BatchCounter(),
                    CSVLogger(f'{tempd}/logs.csv', include_batch_logs=True),
                    ModelCheckpoint(f'{tempd}/epoch{{epoch:03d}}.h5',
                                    save_weights_only=True),
                ],
            )
        except KeyboardInterrupt:
            pass

        with open(f'{tempd}/logs.csv') as fp:
            print(fp.read())


if __name__ == '__main__':
    _main()
