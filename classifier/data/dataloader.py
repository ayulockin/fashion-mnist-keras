import os
import numpy as np
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


class GetDataloader:
    def __init__(self, args):
        self.args = args
        self.train_ds, self.valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])

    def get_dataloader(self, dataloader_type="train"):
        # Select the dataset
        if dataloader_type == "train":
            dataloader = self.train_ds
        else:
            dataloader = self.valid_ds

        # Shuffle if its for training
        if dataloader_type == "train":
            dataloader = dataloader.shuffle(self.args.dataset_config.shuffle_buffer)

        # Load the image
        dataloader = dataloader.map(self.parse_data, num_parallel_calls=AUTOTUNE)

        if self.args.dataset_config.do_cache:
            dataloader = dataloader.cache()

        dataloader = (
            dataloader
            .batch(self.args.dataset_config.batch_size)
        )

        # Add augmentation to dataloader for training
        # if self.args.dataset_config.use_augmentations and dataloader_type=="train":
        #     dataloader = (
        #         dataloader
        #         .map(self.to_dict, num_parallel_calls=AUTOTUNE)
        #         # .map(apply_mixup, num_parallel_calls=AUTOTUNE)
        #         .map(self.to_tuple, num_parallel_calls=AUTOTUNE)
        #     )

        # Prefetch for speedup
        dataloader = dataloader.prefetch(AUTOTUNE)

        return dataloader

    def parse_data(self, example):
        # Get image
        image = example["image"]
        # pixel values from [0, 255] to [0, 1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Get label
        label = example["label"]
        label = tf.cast(label, dtype=tf.int64)
        if self.args.dataset_config.apply_one_hot:
            label = tf.one_hot(label, depth=self.args.dataset_config.num_classes)

        return image, label

    def to_dict(self, images, labels):
        "Apply this after batching"
        return {"images": images, "labels": labels}

    def to_tuple(self, samples_dict):
        "Apply after augmentation is done"
        return samples_dict["images"], samples_dict["labels"]

