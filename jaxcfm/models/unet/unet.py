"""From https://raw.githubusercontent.com/openai/guided-diffusion/main/guided_diffusion/unet.py."""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import (
    checkpoint,
    conv_nd,
    normalization,
    timestep_embedding,
    zero_module,
    silu,
)


class AttentionPool2d(nn.Module):
    """Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py."""

    spacial_dim: int
    embed_dim: int
    num_heads_channels: int
    output_dim: Optional[int] = None

    def setup(self):
        output_dim = self.output_dim or self.embed_dim
        self.positional_embedding = self.param(
            'positional_embedding',
            lambda rng, shape: jax.random.normal(rng, shape) / (self.embed_dim ** 0.5),
            (self.embed_dim, self.spacial_dim ** 2 + 1)
        )
        from .nn import conv_nd
        self.qkv_proj = conv_nd(1, self.embed_dim, 3 * self.embed_dim, 1)
        self.c_proj = conv_nd(1, self.embed_dim, output_dim, 1)
        self.num_heads = self.embed_dim // self.num_heads_channels

    def __call__(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = jnp.concatenate([jnp.mean(x, axis=-1, keepdims=True), x], axis=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].astype(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = QKVAttention(self.num_heads)(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock:
    """Any module where forward() takes timestep embeddings as a second argument."""
    pass


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then upsampling occurs in the
        inner-two dimensions.
    """

    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: Optional[int] = None

    def setup(self):
        # Compute out_channels without modifying fields
        out_channels = self.out_channels or self.channels
        if self.use_conv:
            from .nn import conv_nd
            self.conv = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)

    def __call__(self, x):
        # Note: x might have different channels if it's been processed, so we don't assert
        # assert x.shape[1] == self.channels
        if self.dims == 3:
            # For 3D, upsample the last two dimensions
            x = jax.image.resize(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2, x.shape[4] * 2), method='nearest')
        else:
            # For 1D/2D, use jax.image.resize
            if self.dims == 2:
                x = jax.image.resize(x, (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2), method='nearest')
            else:
                x = jax.image.resize(x, (x.shape[0], x.shape[1], x.shape[2] * 2), method='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then downsampling occurs in the
        inner-two dimensions.
    """

    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: Optional[int] = None

    def setup(self):
        # Compute out_channels without modifying fields
        out_channels = self.out_channels or self.channels
        if self.use_conv:
            from .nn import conv_nd
            if self.dims == 2:
                self.op = conv_nd(self.dims, self.channels, out_channels, 3, strides=2, padding=1)
            elif self.dims == 3:
                self.op = conv_nd(self.dims, self.channels, out_channels, 3, strides=(1, 2, 2), padding=1)
            else:
                self.op = conv_nd(self.dims, self.channels, out_channels, 3, strides=2, padding=1)
        else:
            assert self.channels == self.out_channels
            if self.dims == 2:
                self.op = lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            elif self.dims == 3:
                self.op = lambda x: nn.avg_pool(x, window_shape=(1, 2, 2), strides=(1, 2, 2))
            else:
                self.op = lambda x: nn.avg_pool(x, window_shape=(2,), strides=(2,))

    def __call__(self, x):
        # Note: x might have different channels if it's been processed, so we don't assert
        # assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module, TimestepBlock):
    """A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial convolution instead of a
        smaller 1x1 convolution to change the channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    channels: int
    emb_channels: int
    dropout: float
    out_channels: Optional[int] = None
    use_conv: bool = False
    use_scale_shift_norm: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False

    def setup(self):
        # Compute values without modifying fields (Flax doesn't allow modifying fields in setup)
        out_channels = self.out_channels or self.channels
        updown = self.up or self.down

        # In layers
        self.in_norm = normalization(self.channels)
        from .nn import conv_nd
        self.in_conv = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)

        # Up/down sampling
        if self.up:
            self.h_upd = Upsample(self.channels, False, self.dims)
            self.x_upd = Upsample(self.channels, False, self.dims)
        elif self.down:
            self.h_upd = Downsample(self.channels, False, self.dims)
            self.x_upd = Downsample(self.channels, False, self.dims)

        # Embedding layers
        # Note: emb_out_dim should match the output channels after in_conv
        # Since in_conv outputs out_channels features, emb_out_dim should be based on out_channels
        emb_out_dim = 2 * out_channels if self.use_scale_shift_norm else out_channels
        self.emb_layer1 = lambda x: silu(x)
        self.emb_layer2 = nn.Dense(emb_out_dim)

        # Out layers
        self.out_norm = normalization(out_channels)
        self.out_dropout = nn.Dropout(rate=self.dropout)
        from .nn import conv_nd
        self.out_conv = conv_nd(self.dims, out_channels, out_channels, 3, padding=1)

        # Skip connection
        if out_channels == self.channels:
            self.skip_connection = lambda x: x
        elif self.use_conv:
            from .nn import conv_nd
            self.skip_connection = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)
        else:
            from .nn import conv_nd
            self.skip_connection = conv_nd(self.dims, self.channels, out_channels, 1)

    def __call__(self, x, emb, rng=None):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Array of features.
        :param emb: an [N x emb_channels] Array of timestep embeddings.
        :param rng: optional RNG key for dropout.
        :return: an [N x C x ...] Array of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, emb), (), True)
        else:
            return self._forward(x, emb, rng)

    def _forward(self, x, emb, rng=None):
        updown = self.up or self.down
        if updown:
            h = self.in_norm(x)
            h = silu(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
        else:
            h = self.in_norm(x)
            h = silu(h)
            h = self.in_conv(h)

        emb_out = self.emb_layer1(emb)
        emb_out = self.emb_layer2(emb_out)
        emb_out = emb_out.astype(h.dtype)
        # Add dimensions to match h shape
        # emb_out shape: (batch, emb_dim) -> need to match h shape: (batch, channels, *spatial)
        # In PyTorch, channels are at dim 1, so we add spatial dims after channels
        # emb_out: (batch, emb_dim) -> (batch, emb_dim, 1, 1, ...) to match spatial dims
        # h shape: (batch, channels, *spatial)
        # emb_out should be: (batch, emb_dim, 1, 1, ...) where ... matches spatial dims
        # Ensure batch dimension matches first
        if emb_out.shape[0] != h.shape[0]:
            # If emb has batch=1 but h has batch>1, broadcast emb
            if emb_out.shape[0] == 1:
                emb_out = jnp.broadcast_to(emb_out, (h.shape[0],) + emb_out.shape[1:])
        
        # Check if emb_out's channel dimension matches h's channel dimension
        # If not, we need to adjust (this can happen if there's a dimension mismatch)
        h_channels = h.shape[1]  # Channel dimension of h
        emb_dim = emb_out.shape[1]  # Embedding dimension
        
        if self.use_scale_shift_norm:
            # For scale_shift_norm, emb_out should have 2 * h_channels features
            if emb_dim != 2 * h_channels:
                # Adjust: either pad or truncate
                if emb_dim < 2 * h_channels:
                    padding = jnp.zeros((emb_out.shape[0], 2 * h_channels - emb_dim), dtype=emb_out.dtype)
                    emb_out = jnp.concatenate([emb_out, padding], axis=1)
                else:
                    emb_out = emb_out[:, :2 * h_channels]
            # Add spatial dimensions
            for _ in range(len(h.shape) - 2):  # Add spatial dimensions (skip batch and channel)
                emb_out = emb_out[:, :, None] if len(emb_out.shape) == 2 else emb_out[..., None]
            scale, shift = jnp.split(emb_out, 2, axis=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = silu(h)
            if rng is not None:
                h = self.out_dropout(h, rngs={'dropout': rng})
            else:
                h = self.out_dropout(h, deterministic=True)
            h = self.out_conv(h)
        else:
            # For non-scale_shift_norm, emb_out should have h_channels features
            if emb_dim != h_channels:
                # Adjust: either pad or truncate
                if emb_dim < h_channels:
                    padding = jnp.zeros((emb_out.shape[0], h_channels - emb_dim), dtype=emb_out.dtype)
                    emb_out = jnp.concatenate([emb_out, padding], axis=1)
                else:
                    emb_out = emb_out[:, :h_channels]
            # Add spatial dimensions
            for _ in range(len(h.shape) - 2):  # Add spatial dimensions (skip batch and channel)
                emb_out = emb_out[:, :, None] if len(emb_out.shape) == 2 else emb_out[..., None]
            h = h + emb_out
            h = self.out_norm(h)
            h = silu(h)
            if rng is not None:
                h = self.out_dropout(h, rngs={'dropout': rng})
            else:
                h = self.out_dropout(h, deterministic=True)
            h = self.out_conv(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_checkpoint: bool = False
    use_new_attention_order: bool = False

    def setup(self):
        # Compute num_heads_val
        if self.num_head_channels == -1:
            num_heads_val = self.num_heads
        else:
            assert (
                self.channels % self.num_head_channels == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_head_channels {self.num_head_channels}"
            num_heads_val = self.channels // self.num_head_channels
        
        # Ensure channels is divisible by num_heads_val for QKV attention
        # The qkv output has shape (channels * 3), and we need (channels * 3) % (3 * num_heads_val) == 0
        # which simplifies to channels % num_heads_val == 0
        if self.channels % num_heads_val != 0:
            # Adjust num_heads_val to be a divisor of channels
            # Find the largest divisor of channels that is <= num_heads_val
            for n in range(num_heads_val, 0, -1):
                if self.channels % n == 0:
                    num_heads_val = n
                    break
            else:
                raise ValueError(
                    f"channels {self.channels} must be divisible by num_heads_val. "
                    f"Got num_heads_val={num_heads_val} (from num_heads={self.num_heads} or num_head_channels={self.num_head_channels})"
                )
        
        # Store as attribute (not a field, so this is OK)
        self.num_heads_val = num_heads_val

        # Create normalization with self.channels
        # Note: In Flax, normalization will adapt to actual input channels
        self.norm = normalization(self.channels)
        # Use conv_nd for 1D conv to match PyTorch behavior
        from .nn import conv_nd
        # Create qkv with self.channels - Flax Conv will adapt to input channels
        self.qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        if self.use_new_attention_order:
            self.attention = QKVAttention(num_heads_val)
        else:
            self.attention = QKVAttentionLegacy(num_heads_val)

        # Zero module for proj_out (outputs self.channels features to match input)
        from .nn import zero_module, conv_nd
        self.proj_out = zero_module(conv_nd(1, self.channels, self.channels, 1))

    def __call__(self, x, rng=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x,), (), True)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        # Match PyTorch behavior: assume c == self.channels
        # In the UNet, AttentionBlock is always created with channels matching the input
        x_flat = x.reshape(b, c, -1)  # Shape: (b, c, spatial_size)
        x_norm = self.norm(x_flat)  # Shape: (b, self.channels, spatial_size)
        qkv = self.qkv(x_norm)  # Shape: (b, self.channels*3, spatial_size)
        h = self.attention(qkv)  # Should return (b, self.channels, spatial_size)
        h = self.proj_out(h)  # (b, self.channels, spatial_size)
        return (x_flat + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention.

    Matches legacy QKVAttention + input/output heads shaping
    """

    n_heads: int

    def __call__(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        # qkv has shape (bs, channels*3, spatial_size)
        # We need width % (3 * n_heads) == 0, which means (channels * 3) % (3 * n_heads) == 0
        # This simplifies to channels % n_heads == 0
        channels = width // 3
        if channels % self.n_heads != 0:
            # Find compatible n_heads that divides channels
            compatible_heads = 1
            for n in range(1, min(self.n_heads, channels) + 1):
                if channels % n == 0:
                    compatible_heads = n
            n_heads_used = compatible_heads
        else:
            n_heads_used = self.n_heads
        ch = channels // n_heads_used
        
        qkv_reshaped = qkv.reshape(bs * n_heads_used, ch * 3, length)
        q, k, v = jnp.split(qkv_reshaped, 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        a = jnp.einsum("bts,bcs->bct", weight, v)
        # Reshape back: (bs*n_heads_used, ch, length) -> (bs, n_heads_used*ch, length)
        a_reshaped = a.reshape(bs, n_heads_used * ch, length)
        return a_reshaped


class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different order."""

    n_heads: int

    def __call__(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        # width should be channels * 3
        # We need width % (3 * n_heads) == 0
        if width % (3 * self.n_heads) != 0:
            # Find compatible n_heads
            width_per_head = width // 3
            compatible_heads = 1
            for n in range(1, min(self.n_heads, width_per_head) + 1):
                if width_per_head % n == 0:
                    compatible_heads = n
            n_heads_used = compatible_heads
        else:
            n_heads_used = self.n_heads
        
        ch = width // (3 * n_heads_used)
        # Split qkv: (bs, width, length) where width = channels*3
        # Split into q, k, v: each should have width/3 = channels features
        q, k, v = jnp.split(qkv, 3, axis=1)  # Each: (bs, width/3, length) = (bs, channels, length)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # Reshape each: (bs, channels, length) -> (bs*n_heads_used, ch, length)
        # where channels = n_heads_used * ch
        q_reshaped = (q * scale).reshape(bs * n_heads_used, ch, length)
        k_reshaped = (k * scale).reshape(bs * n_heads_used, ch, length)
        v_reshaped = v.reshape(bs * n_heads_used, ch, length)
        weight = jnp.einsum("bct,bcs->bts", q_reshaped, k_reshaped)
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        a = jnp.einsum("bts,bcs->bct", weight, v_reshaped)
        # Reshape back: (bs*n_heads_used, ch, length) -> (bs, n_heads_used*ch, length)
        # This should equal (bs, channels, length) where channels = width/3
        a_reshaped = a.reshape(bs, n_heads_used * ch, length)
        return a_reshaped


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which attention will take
        place. May be a set, list, or tuple. For example, if this contains 4, then at 4x
        downsampling, attention will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be class-conditional with
        `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_head_channels: if specified, ignore num_heads and instead use a fixed channel width
        per attention head.
    :param num_heads_upsample: works with num_heads to set a different number of heads for
        upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially increased
        efficiency.
    """

    image_size: int
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: tuple
    dropout: float = 0
    channel_mult: tuple = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False

    def setup(self):
        if self.num_heads_upsample == -1:
            num_heads_upsample = self.num_heads
        else:
            num_heads_upsample = self.num_heads_upsample

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential([
            nn.Dense(time_embed_dim),
            lambda x: silu(x),
            nn.Dense(time_embed_dim),
        ])

        if self.num_classes is not None:
            self.label_emb = nn.Embed(self.num_classes, time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        
        # Input blocks - build list first, then convert to tuple
        # First block is a conv that changes in_channels to ch
        from .nn import conv_nd
        input_blocks_list = []
        first_conv = conv_nd(self.dims, self.in_channels, ch, 3, padding=1)
        # Wrap in a tuple to match the structure of other blocks
        input_blocks_list.append((first_conv,))

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                # Convert layers list to tuple (Flax best practice)
                input_blocks_list.append(tuple(layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                if self.resblock_updown:
                    down_block = ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=out_ch,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        down=True,
                    )
                else:
                    down_block = Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                # Convert to tuple (Flax best practice)
                input_blocks_list.append((down_block,))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        # Convert input_blocks to tuple (Flax best practice)
        self.input_blocks = tuple(input_blocks_list)

        # Middle block - use tuple (Flax best practice)
        self.middle_block = (
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            AttentionBlock(
                channels=ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Output blocks - build list first, then convert to tuple
        output_blocks_list = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    if self.resblock_updown:
                        up_block = ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                    else:
                        up_block = Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    layers.append(up_block)
                    ds //= 2
                # Convert layers list to tuple (Flax best practice)
                output_blocks_list.append(tuple(layers))
                self._feature_size += ch
        
        # Convert output_blocks to tuple (Flax best practice)
        self.output_blocks = tuple(output_blocks_list)

        # Output layer
        from .nn import conv_nd
        self.out_norm = normalization(ch)
        self.out_conv = conv_nd(self.dims, ch, self.out_channels, 3, padding=1)

    def __call__(self, t, x, y=None, rng=None):
        """Apply the model to an input batch.

        :param t: timesteps, a 1-D batch of timesteps.
        :param x: an [N x C x ...] Array of inputs.
        :param y: an [N] Array of labels, if class-conditional.
        :param rng: optional RNG key for dropout.
        :return: an [N x C x ...] Array of outputs.
        """
        timesteps = t
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        # Handle timesteps shape
        while timesteps.ndim > 1:
            timesteps = timesteps[:, 0]
        if timesteps.ndim == 0:
            timesteps = jnp.repeat(timesteps[None], x.shape[0])

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.astype(jnp.float16 if self.use_fp16 else jnp.float32)
        hs = []
        
        # Input blocks
        for block in self.input_blocks:
            if isinstance(block, (list, tuple)):
                for layer in block:
                    if isinstance(layer, TimestepBlock):
                        h = layer(h, emb, rng)
                    else:
                        h = layer(h)
            else:
                h = block(h)
            hs.append(h)
        
        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, TimestepBlock):
                h = layer(h, emb, rng)
            else:
                h = layer(h)
        
        # Output blocks
        for block in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=1)
            for layer in block:
                if isinstance(layer, TimestepBlock):
                    h = layer(h, emb, rng)
                else:
                    h = layer(h)
        
        h = h.astype(x.dtype)
        h = self.out_norm(h)
        h = silu(h)
        h = self.out_conv(h)
        return h


NUM_CLASSES = 1000


def create_unet_model(
    dim,
    num_channels,
    num_res_blocks,
    channel_mult=None,
    learn_sigma=False,
    class_cond=False,
    num_classes=NUM_CLASSES,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    **kwargs
):
    """Factory function to create UNetModel with convenient interface.
    
    Dim (tuple): (C, H, W)
    """
    image_size = dim[-1]
    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 28:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(channel_mult) if not isinstance(channel_mult, tuple) else channel_mult

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=dim[0],
        model_channels=num_channels,
        out_channels=(dim[0] if not learn_sigma else dim[0] * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        **kwargs
    )


# For backward compatibility, create an alias
class UNetModelWrapper:
    """Wrapper class for backward compatibility. Use create_unet_model() instead."""
    def __init__(self, *args, **kwargs):
        self._model = create_unet_model(*args, **kwargs)
    
    def init(self, rng, *args, **kwargs):
        return self._model.init(rng, *args, **kwargs)
    
    def apply(self, params, *args, **kwargs):
        return self._model.apply(params, *args, **kwargs)

