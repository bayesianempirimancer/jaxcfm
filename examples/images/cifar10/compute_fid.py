# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import sys

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from absl import flags
from cleanfid import fid

from jaxcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")

FLAGS(sys.argv)


# Define the model
# In JAX/Flax, models need proper initialization
# This is a placeholder - actual usage would need:
# - Proper model initialization
# - Parameter loading from checkpoint
key = random.PRNGKey(42)

# new_net = UNetModelWrapper(...)
# params = new_net.init(key, ...)

# Load the model
# PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
# checkpoint = ...  # Load from JAX checkpoint format
# params = checkpoint["ema_model"]


def gen_1_img(key, unused_latent):
    """Generate one image using the model."""
    key, subkey = random.split(key)
    x = random.normal(subkey, (FLAGS.batch_size_fid, 3, 32, 32))
    
    # ODE integration would go here
    # Using a simple Euler method as placeholder
    if FLAGS.integration_method == "euler":
        print("Use method: ", FLAGS.integration_method)
        dt = 1.0 / FLAGS.integration_steps
        for i in range(FLAGS.integration_steps):
            t = i * dt
            # Apply model
            # v = model.apply(params, t, x)
            # x = x + dt * v
            pass
    else:
        print("Use method: ", FLAGS.integration_method)
        # Use JAX ODE solver (would need diffrax or similar)
        # from diffrax import diffeqsolve, ODETerm, Tsit5
        # term = ODETerm(model.apply)
        # solution = diffeqsolve(term, ...)
        pass
    
    # Convert to uint8
    img = (x * 127.5 + 128).clip(0, 255).astype(jnp.uint8)
    return np.array(img)


def main():
    """Main function to compute FID."""
    print("Start computing FID")
    # Note: FID computation would need proper key management
    key = random.PRNGKey(42)
    score = fid.compute_fid(
        gen=lambda unused: gen_1_img(key, unused),
        dataset_name="cifar10",
        batch_size=FLAGS.batch_size_fid,
        dataset_res=32,
        num_gen=FLAGS.num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )
    print()
    print("FID has been computed")
    print()
    print("FID: ", score)


if __name__ == "__main__":
    main()

