# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import jax
import jax.numpy as jnp
from absl import app, flags
from jax import random
from tqdm import trange
import numpy as np
import torch
from torchvision import datasets, transforms
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

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

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
    # In JAX, we typically use numpy arrays
    # Note: For JAX, you might want to use a different data loading approach
    # This keeps torch DataLoader for compatibility but converts to JAX arrays
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS - In JAX/Flax, models need to be initialized differently
    # This is a simplified version - actual usage would need proper Flax initialization
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    
    # Note: UNetModelWrapper in JAX would need proper initialization
    # For now, this is a placeholder structure
    # net_model = UNetModelWrapper(...)
    # ema_model = copy.deepcopy(net_model)  # In JAX, this would copy parameters
    
    # Optimizer - In JAX, we use optax
    import optax
    # optim = optax.adam(learning_rate=FLAGS.lr)
    # sched = optax.linear_schedule(...)  # Learning rate schedule
    
    # Note: JAX training loop would be significantly different
    # This is a structural conversion - actual implementation would need:
    # - Proper model initialization with Flax
    # - JAX-compatible data loading
    # - JAX training step function with jit compilation
    # - Different checkpointing mechanism

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

    # JAX training loop would be structured differently
    # This is a placeholder - actual implementation would need:
    # - @jax.jit decorated training step
    # - Proper parameter updates using optax
    # - JAX-compatible checkpointing
    
    print("JAX version - training loop implementation needed")
    print("This requires proper Flax model initialization and JAX training infrastructure")


if __name__ == "__main__":
    app.run(train)

