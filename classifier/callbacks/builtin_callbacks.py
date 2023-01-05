import os

import tensorflow as tf

import wandb


def get_earlystopper(args):
    args = args.callback_config.early_stopper

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor=args.monitor,
        patience=args.early_patience,
        verbose=0,
        mode="auto",
        restore_best_weights=args.restore_best_weights,
    )

    return earlystopper


def get_reduce_lr_on_plateau(args):
    args = args.callback_config.reduce_lr_on_plateau

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=args.monitor, factor=args.factor, patience=args.patience
    )

    return reduce_lr_on_plateau
