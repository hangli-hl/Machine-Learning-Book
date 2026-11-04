##############################################################################
#          U-SHAPED NETWORKS (U_NET) FOR DDPM / SMLD / FLOW MATCHING
##############################################################################

import math

import torch
import torch.nn as nn


##############################################################################
#                           2D UNET BUILDING BLOCKS
##############################################################################


class ConditionEmbedding(nn.Module):
    """
    Embedding block for a scalar condition (diffusion timestep, noise level
    or continuous flow-matching time).

    The block consists of:
    - Optional rescaling of the condition by condition_scale
    - Sinusoidal encoding to a vector of embedding_size
    - Hidden layer with SiLU activation, then an output layer

    Continuous times in [0, 1] (flow matching) must be rescaled before the
    sinusoidal encoding, otherwise they only excite the lowest frequency and
    the network becomes nearly blind to time. Integer timesteps / noise
    indices use condition_scale = 1.0.

    Input: (batch_size,)
    Output: (batch_size, output_size)

    Attributes:
        embedding_size: size of the sinusoidal encoding, a positive even integer
        condition_scale: factor rescaling the condition, a positive float
        mlp: sequential container of network layers
    """

    def __init__(self, embedding_size, output_size, condition_scale=1.0):
        """
        Initialize a condition embedding block.

        Args:
            embedding_size: size of the sinusoidal encoding, a positive even integer
            output_size: size of the output embedding, a positive integer
            condition_scale: factor rescaling the condition, a positive float
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.condition_scale = condition_scale

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, conditions):
        """
        Forward pass: embed the scalar conditions.

        Args:
            conditions: conditioning scalars of shape (batch_size,)

        Returns:
            embeddings: condition embeddings, shape (batch_size, output_size)
        """
        conditions = conditions.float().reshape(-1) * self.condition_scale

        half_size = self.embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_size, device=conditions.device).float()
            / half_size
        )
        angles = conditions.reshape(-1, 1) * frequencies.reshape(1, -1)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        embeddings = self.mlp(encodings)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block conditioned on the condition embedding.

    The block consists of:
    - Convolution path 1: GroupNorm, SiLU activation and 3x3 convolution
    - Condition projection: linear layer adding the embedding to the features
    - Convolution path 2: GroupNorm, SiLU activation and 3x3 convolution
    - Skip connection: identity or 1x1 convolution when channels change

    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height, width)

    Attributes:
        norm1: group normalization before the first convolution
        conv1: first 3x3 convolution
        condition_projection: linear layer projecting the condition embedding
        norm2: group normalization before the second convolution
        conv2: second 3x3 convolution
        skip: skip connection layer
        activation: SiLU activation
    """

    def __init__(
        self, in_channels, out_channels, condition_embedding_size, num_groups=8
    ):
        """
        Initialize a residual block.

        Args:
            in_channels: number of input channels, a positive integer
            out_channels: number of output channels, a positive integer
            condition_embedding_size: size of the condition embedding, a positive integer
            num_groups: number of groups for group normalization, a positive integer
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Linear(condition_embedding_size, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, features, condition_embeddings):
        """
        Forward pass: refine features conditioned on the condition embedding.

        Args:
            features: input features of shape (batch_size, in_channels, height, width)
            condition_embeddings: embeddings of shape (batch_size, condition_embedding_size)

        Returns:
            outputs: output features, shape (batch_size, out_channels, height, width)
        """
        hiddens = self.conv1(self.activation(self.norm1(features)))
        hiddens = (
            hiddens
            + self.condition_projection(self.activation(condition_embeddings))[
                :, :, None, None
            ]
        )
        hiddens = self.conv2(self.activation(self.norm2(hiddens)))

        outputs = hiddens + self.skip(features)
        return outputs


class AttentionBlock(nn.Module):
    """
    Single-head self-attention block over spatial positions.

    The block consists of:
    - GroupNorm
    - 1x1 convolution producing queries, keys and values
    - Scaled dot-product attention over the height * width positions
    - 1x1 output projection and a residual connection

    Input: (batch_size, channels, height, width)
    Output: (batch_size, channels, height, width)

    Attributes:
        norm: group normalization before the attention
        qkv: 1x1 convolution producing queries, keys and values
        projection: 1x1 output convolution
    """

    def __init__(self, channels, num_groups=8):
        """
        Initialize an attention block.

        Args:
            channels: number of input and output channels, a positive integer
            num_groups: number of groups for group normalization, a positive integer
        """
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, features):
        """
        Forward pass: mix features across all spatial positions.

        Args:
            features: input features of shape (batch_size, channels, height, width)

        Returns:
            outputs: output features, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = features.shape

        queries, keys, values = (
            self.qkv(self.norm(features))
            .reshape(batch_size, 3, channels, height * width)
            .unbind(1)
        )
        weights = torch.softmax(
            queries.transpose(1, 2) @ keys / math.sqrt(channels), dim=-1
        )
        hiddens = (values @ weights.transpose(1, 2)).reshape(
            batch_size, channels, height, width
        )

        outputs = features + self.projection(hiddens)
        return outputs


class DownBlock(nn.Module):
    """
    Downsampling block halving the spatial resolution.

    The block consists of:
    - Downsampling: 3x3 convolution with stride 2
    - Residual block conditioned on the condition embedding

    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height / 2, width / 2)

    Attributes:
        down: strided convolution for downsampling
        block: residual block
    """

    def __init__(self, in_channels, out_channels, condition_embedding_size):
        """
        Initialize a downsampling block.

        Args:
            in_channels: number of input channels, a positive integer
            out_channels: number of output channels, a positive integer
            condition_embedding_size: size of the condition embedding, a positive integer
        """
        super().__init__()

        self.down = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
        self.block = ResidualBlock(in_channels, out_channels, condition_embedding_size)

    def forward(self, features, condition_embeddings):
        """
        Forward pass: downsample and refine features.

        Args:
            features: input features of shape (batch_size, in_channels, height, width)
            condition_embeddings: embeddings of shape (batch_size, condition_embedding_size)

        Returns:
            outputs: output features, shape (batch_size, out_channels, height / 2, width / 2)
        """
        outputs = self.block(self.down(features), condition_embeddings)
        return outputs


class UpBlock(nn.Module):
    """
    Upsampling block doubling the spatial resolution.

    The block consists of:
    - Upsampling: nearest-neighbor interpolation followed by a 3x3 convolution
    - Concatenation with the skip connection from the encoder
    - Residual block conditioned on the condition embedding

    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height * 2, width * 2)

    Attributes:
        up: sequential container of the upsampling layers
        block: residual block
    """

    def __init__(
        self, in_channels, skip_channels, out_channels, condition_embedding_size
    ):
        """
        Initialize an upsampling block.

        Args:
            in_channels: number of input channels, a positive integer
            skip_channels: number of channels of the skip connection, a positive integer
            out_channels: number of output channels, a positive integer
            condition_embedding_size: size of the condition embedding, a positive integer
        """
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.block = ResidualBlock(
            in_channels + skip_channels, out_channels, condition_embedding_size
        )

    def forward(self, features, skip_features, condition_embeddings):
        """
        Forward pass: upsample, merge the skip connection and refine features.

        Args:
            features: input features of shape (batch_size, in_channels, height, width)
            skip_features: encoder features of shape
                (batch_size, skip_channels, height * 2, width * 2)
            condition_embeddings: embeddings of shape (batch_size, condition_embedding_size)

        Returns:
            outputs: output features, shape (batch_size, out_channels, height * 2, width * 2)
        """
        features = self.up(features)
        outputs = self.block(
            torch.cat([skip_features, features], dim=1), condition_embeddings
        )
        return outputs


##############################################################################
#                           2D UNET (MNIST)
##############################################################################


class UNet(nn.Module):
    """
    U-shaped network for MNIST generative models (DDPM, SMLD, Flow matching).

    The output is the raw predicted field:
    - DDPM: predicted noise, condition = integer timestep
    - SMLD: predicted noise level, condition = integer noise level
    - Flow matching: predicted velocity, condition = continuous time in [0, 1]
                       (use condition_scale = 1000.0)

    It consists of:
    - Condition embedding: sinusoidal encoding followed by a two-layer MLP
    - Stem: 3x3 convolution mapping images to base_channels feature maps
    - Encoder level 1: residual block at 28x28 with base_channels channels
    - Encoder level 2: downsampling block at 14x14 with 2 * base_channels channels
    - Encoder level 3: downsampling block at 7x7 with 4 * base_channels channels
    - Bottleneck: residual block, attention block and residual block at 7x7
    - Decoder level 1: upsampling block at 14x14 with 2 * base_channels channels
    - Decoder level 2: upsampling block at 28x28 with base_channels channels
    - Output layer: GroupNorm, SiLU activation and 3x3 convolution to image_channels

    Input: (batch_size, image_channels, image_size, image_size) and (batch_size,)
    Output: (batch_size, image_channels, image_size, image_size)

    Attributes:
        image_channels: number of channels of input images, a positive integer
        image_size: size of input images (height = width), a positive integer
        condition_embedding: condition embedding block
        stem: input convolution
        encoder1: residual block at the full resolution
        encoder2: first downsampling block
        encoder3: second downsampling block
        middle1: first bottleneck residual block
        middle_attention: bottleneck attention block
        middle2: second bottleneck residual block
        decoder1: first upsampling block
        decoder2: second upsampling block
        output_layer: sequential container of the output layers
    """

    def __init__(
        self, image_channels=1, image_size=28, base_channels=64, condition_scale=1.0
    ):
        """
        Initialize a 2D UNet.

        Args:
            image_channels: number of channels of input images, a positive integer
            image_size: size of input images (height = width), a positive integer
            base_channels: number of channels at the full resolution,
                a positive integer that is a multiple of 8
            condition_scale: factor rescaling the condition before the sinusoidal
                encoding (1.0 for integer timesteps / noise indices, e.g. 1000.0
                for continuous flow-matching times)
        """
        super().__init__()
        assert image_size == 28, "The UNet architecture is designed for 28x28 images"

        self.image_channels = image_channels
        self.image_size = image_size

        channels1 = base_channels
        channels2 = base_channels * 2
        channels3 = base_channels * 4
        condition_embedding_size = base_channels * 4

        self.condition_embedding = ConditionEmbedding(
            base_channels, condition_embedding_size, condition_scale
        )

        self.stem = nn.Conv2d(image_channels, channels1, kernel_size=3, padding=1)
        self.encoder1 = ResidualBlock(channels1, channels1, condition_embedding_size)
        self.encoder2 = DownBlock(channels1, channels2, condition_embedding_size)
        self.encoder3 = DownBlock(channels2, channels3, condition_embedding_size)

        self.middle1 = ResidualBlock(channels3, channels3, condition_embedding_size)
        self.middle_attention = AttentionBlock(channels3)
        self.middle2 = ResidualBlock(channels3, channels3, condition_embedding_size)

        self.decoder1 = UpBlock(
            channels3, channels2, channels2, condition_embedding_size
        )
        self.decoder2 = UpBlock(
            channels2, channels1, channels1, condition_embedding_size
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(8, channels1),
            nn.SiLU(),
            nn.Conv2d(channels1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, images, conditions):
        """
        Forward pass: predict the raw field (noise or velocity).

        Args:
            images: images of shape (batch_size, image_channels, image_size, image_size)
            conditions: conditioning scalars of shape (batch_size,)

        Returns:
            outputs: predicted field, shape (batch_size, image_channels, image_size, image_size)
        """
        condition_embeddings = self.condition_embedding(conditions)

        features1 = self.encoder1(self.stem(images), condition_embeddings)
        features2 = self.encoder2(features1, condition_embeddings)
        features3 = self.encoder3(features2, condition_embeddings)

        hiddens = self.middle1(features3, condition_embeddings)
        hiddens = self.middle_attention(hiddens)
        hiddens = self.middle2(hiddens, condition_embeddings)

        hiddens = self.decoder1(hiddens, features2, condition_embeddings)
        hiddens = self.decoder2(hiddens, features1, condition_embeddings)

        outputs = self.output_layer(hiddens)
        return outputs


##############################################################################
#          1D NETWORK BUILDING BLOCKS (GAUSSIAN MIXTURE TOY DATA)
##############################################################################


class ConditionEmbedding1D(nn.Module):
    """
    Embedding block for a scalar condition (diffusion timestep, noise level
    index or continuous flow-matching time) in the 1D toy models.

    The block consists of:
    - Optional rescaling of the condition by condition_scale
    - Sinusoidal encoding to a vector of embedding_size
    - Hidden layer with SiLU activation, then an output layer

    Continuous times in [0, 1] (flow matching) must be rescaled before the
    sinusoidal encoding, otherwise they only excite the lowest frequency and
    the network becomes nearly blind to time. Integer timesteps / noise
    indices use condition_scale = 1.0.

    Input: (batch_size,)
    Output: (batch_size, embedding_size)

    Attributes:
        embedding_size: size of the sinusoidal encoding, a positive even integer
        condition_scale: factor rescaling the condition, a positive float
        mlp: sequential container of network layers
    """

    def __init__(self, embedding_size, condition_scale=1.0):
        """
        Initialize a 1D condition embedding block.

        Args:
            embedding_size: size of the sinusoidal encoding, a positive even integer
            condition_scale: factor rescaling the condition, a positive float
        """
        super().__init__()
        assert embedding_size % 2 == 0, "embedding_size must be even"

        self.embedding_size = embedding_size
        self.condition_scale = condition_scale

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, conditions):
        """
        Forward pass: embed the scalar conditions.

        Args:
            conditions: conditioning scalars of shape (batch_size,)

        Returns:
            embeddings: condition embeddings, shape (batch_size, embedding_size)
        """
        conditions = conditions.float().reshape(-1) * self.condition_scale

        half_size = self.embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_size, device=conditions.device).float()
            / half_size
        )
        angles = conditions.reshape(-1, 1) * frequencies.reshape(1, -1)
        encodings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        embeddings = self.mlp(encodings)
        return embeddings


##############################################################################
#                1D UNET (GAUSSIAN MIXTURE TOY DATA)
##############################################################################


class UNet1D(nn.Module):
    """
    U-shaped network for the 1D toy generative models (DDPM noise prediction,
    SMLD score prediction or flow-matching velocity prediction).

    The output is the raw predicted field:
    - DDPM           : predicted noise, condition = integer timestep
                       (condition_scale = 1.0)
    - Flow matching  : predicted velocity, condition = continuous time in [0, 1]
                       (condition_scale = 1000.0, zero_init_output = True)

    It consists of:
    - Condition embedding: sinusoidal encoding followed by a two-layer MLP
    - Stem: linear layer mapping the scalar input to hidden_size features
    - Encoder: linear layer with its own condition projection, then dropout
    - Decoder: linear layer fed by the encoder output plus the stem skip connection
    - Output layer: linear layer to a single value

    Input: (batch_size, 1) and (batch_size,)
    Output: (batch_size, 1)

    Attributes:
        activation: SiLU activation
        condition_embedding: condition embedding block
        stem: input linear layer
        encoder: hidden linear layer
        condition_projection: linear layer projecting the condition embedding
        dropout: dropout layer for regularization
        decoder: linear layer merging the skip connection
        output_layer: output linear layer
    """

    def __init__(
        self,
        hidden_size=128,
        condition_scale=1.0,
        dropout=0.1,
        zero_init_output=False,
    ):
        """
        Initialize a 1D UNet.

        Args:
            hidden_size: number of hidden units, a positive even integer
            condition_scale: factor rescaling the condition, a positive float
            dropout: dropout probability, a float in [0, 1)
            zero_init_output: if True, zero-initialize the output layer (used by
                flow matching so the flow starts as the identity map)
        """
        super().__init__()

        self.activation = nn.SiLU()
        self.condition_embedding = ConditionEmbedding1D(hidden_size, condition_scale)

        self.stem = nn.Linear(1, hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)
        self.condition_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, conditions):
        """
        Forward pass: predict the raw field (noise, score or velocity).

        Args:
            x: current states of shape (batch_size, 1)
            conditions: conditioning scalars of shape (batch_size,)

        Returns:
            outputs: predicted field, shape (batch_size, 1)
        """
        condition_embeddings = self.condition_embedding(conditions)

        hiddens1 = self.activation(self.stem(x) + condition_embeddings)
        hiddens1 = self.dropout(hiddens1)
        hiddens2 = self.activation(
            self.encoder(hiddens1) + self.condition_projection(condition_embeddings)
        )
        hiddens2 = self.dropout(hiddens2)

        hiddens3 = self.activation(self.decoder(hiddens2 + hiddens1))

        outputs = self.output_layer(hiddens3)
        return outputs
