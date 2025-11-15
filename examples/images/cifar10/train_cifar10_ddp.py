# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong
#          Imahn Shekhzadeh

import copy
import math
import os

import jax
import jax.numpy as jnp
from absl import app, flags
from jax import random
import numpy as np
from tqdm import trange
from torchvision import datasets, transforms
import torch
try:
    from .utils_cifar_jax import ema, generate_samples, infiniteloop
except ImportError:
    from utils_cifar_jax import ema, generate_samples, infiniteloop

from jaxcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from jaxcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string(
    "master_addr", "localhost", help="master address for Distributed Data Parallel"
)
flags.DEFINE_string("master_port", "12355", help="master port for Distributed Data Parallel")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(rank, total_num_gpus, argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # Note: JAX distributed training uses different mechanisms than PyTorch
    # JAX uses jax.devices() and pmap/pjit for multi-device training
    # This is a structural conversion - actual implementation would need:
    # - JAX device management
    # - pmap or pjit for distributed training
    # - JAX-compatible data loading

    if FLAGS.parallel and total_num_gpus > 1:
        # JAX uses different distributed mechanisms
        # devices = jax.devices()
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    # Convert to JAX-compatible data loading
    import torch
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)

    # MODELS - In JAX/Flax, models need proper initialization
    key = random.PRNGKey(42)
    # net_model initialization would go here
    # ema_model would be a copy of parameters

    # Optimizer - In JAX, we use optax
    import optax
    # optim = optax.adam(learning_rate=FLAGS.lr)
    # sched = optax.linear_schedule(...)

    # show model size
    # model_size = sum(x.size for x in jax.tree_leaves(net_model))
    # print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    global_step = 0  # to keep track of the global step in training loop

    # JAX training loop would be structured differently
    # This is a placeholder - actual implementation would need:
    # - @jax.jit decorated training step
    # - Proper parameter updates using optax
    # - JAX-compatible checkpointing
    # - Multi-device training with pmap/pjit
    
    print("JAX version - training loop implementation needed")
    print("This requires proper Flax model initialization and JAX training infrastructure")


def main(argv):
    # JAX device management
    # devices = jax.devices()
    # total_num_gpus = len(devices)
    total_num_gpus = 1  # Placeholder

    if FLAGS.parallel and total_num_gpus > 1:
        # JAX distributed training setup
        train(rank=0, total_num_gpus=total_num_gpus, argv=argv)
    else:
        train(rank=0, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)

