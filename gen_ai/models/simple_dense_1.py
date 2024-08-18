from typing import Tuple, List
from tensorflow.keras import layers, models
from dataclasses import dataclass


@dataclass
class Config:
    input_shape: Tuple
    dense_layer_sizes: List[int]
    dense_layer_activation: str
    output_classes: int
    output_activation: str


def make_model(config: Config):
    ## 2. Build the model <a name="build"></a>
    input_layer = layers.Input(config.input_shape)

    x = layers.Flatten()(input_layer)
    for dense_layer_size in config.dense_layer_sizes:
        x = layers.Dense(dense_layer_size, activation=config.dense_layer_activation)(x)

    output_layer = layers.Dense(
        config.output_classes, activation=config.output_activation
    )(x)

    return models.Model(input_layer, output_layer)
