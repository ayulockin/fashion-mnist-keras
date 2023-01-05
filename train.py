import tensorflow as tf

from absl import app, flags
from ml_collections.config_flags import config_flags

# Modules
from classifier.data import GetDataloader


FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)

print(CONFIG)

def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    # CALLBACKS = []
    # # Initialize a Weights and Biases run.
    # if FLAGS.wandb:
    #     run = wandb.init(
    #         entity=CONFIG.value.wandb_config.entity,
    #         project=CONFIG.value.wandb_config.project,
    #         job_type="train",
    #         config=config.to_dict(),
    #     )
    #     # Initialize W&B metrics logger callback.
    #     CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Build dataloaders
    dataloader = GetDataloader(config)
    trainloader = dataloader.get_dataloader(dataloader_type="train")
    validloader = dataloader.get_dataloader(dataloader_type="valid")

    # # Initialize callbacks
    # callback_config = config.callback_config
    # # Builtin early stopping callback
    # if callback_config.use_earlystopping:
    #     earlystopper = callbacks.get_earlystopper(config)
    #     CALLBACKS += [earlystopper]
    # # Built in callback to reduce learning rate on plateau
    # if callback_config.use_reduce_lr_on_plateau:
    #     reduce_lr_on_plateau = callbacks.get_reduce_lr_on_plateau(config)
    #     CALLBACKS += [reduce_lr_on_plateau]

    # # Initialize Model checkpointing callback
    # if FLAGS.log_model:
    #     # Custom W&B model checkpoint callback
    #     model_checkpointer = callbacks.get_model_checkpoint_callback(config)
    #     CALLBACKS += [model_checkpointer]

    # if wandb.run is not None:
    #     if FLAGS.log_eval:
    #         model_pred_viz = callbacks.get_evaluation_callback(config, validloader)
    #         CALLBACKS += [model_pred_viz]

    # if callback_config.use_tensorboard:
    #     CALLBACKS += [tf.keras.callbacks.TensorBoard()]

    # # Build the model
    # tf.keras.backend.clear_session()
    # model = SimpleSupervisedModel(config).get_model()
    # model.summary()

    # # Build the pipeline
    # pipeline = SupervisedPipeline(model, config, class_weights, CALLBACKS)

    # # Train and Evaluate
    # pipeline.train_and_evaluate(valid_df, trainloader, validloader)


if __name__ == "__main__":
    app.run(main)