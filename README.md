# SigRNN: Signature-Enhanced Recurrent Neural Networks

This repository implements novel RNN layers that incorporate path signatures for enhanced time series modeling. The implementation is built on top of [keras_sig](https://github.com/remigenet/keras_sig) and is compatible with Keras 3.0, supporting multiple backends (TensorFlow, JAX, and PyTorch).

## Overview

SigRNN introduces two novel layer architectures:
- **SignatureLSTM**: An LSTM variant where the forget gate is computed using path signatures
- **SignatureGRU**: A GRU variant where the reset gate is computed using path signatures

The key idea is to leverage path signatures to enhance the gating mechanisms in traditional RNN architectures. The signature computations provide a richer representation of the temporal dynamics, potentially improving the model's ability to capture long-term dependencies.

## Installation

```bash
pip install sig_rnn
```

## Quick Start

```python
from sig_rnn import SignatureLSTM, SignatureGRU
import keras

# Example with SignatureLSTM
model = keras.Sequential([
    keras.layers.Input(shape=(sequence_length, n_features)),
    SignatureLSTM(
        units=64,
        signature_depth=2,
        signature_input_size=5,
        return_sequences=True
    ),
    keras.layers.Dense(1)
])

# Example with SignatureGRU
model = keras.Sequential([
    keras.layers.Input(shape=(sequence_length, n_features)),
    SignatureGRU(
        units=64,
        signature_depth=2,
        signature_input_size=5,
        return_sequences=False
    ),
    keras.layers.Dense(1)
])
```

## Layer Parameters

### Common Parameters
- `units`: Dimensionality of the output space
- `signature_depth`: Maximum depth for signature computation (default: 2)
- `signature_input_size`: Input dimension for signature computation (default: 5)
- `return_sequences`: Whether to return the full sequence or just the last output
- `return_state`: Whether to return states in addition to output
- `unroll_level`: Level of unrolling for scan operations (default: 10)

### SignatureLSTM
The SignatureLSTM modifies the standard LSTM by computing the forget gate using path signatures while maintaining the traditional computation for input, cell, and output gates. This allows the model to potentially capture more complex temporal patterns when deciding what information to forget.

### SignatureGRU
The SignatureGRU modifies the standard GRU by computing the reset gate using path signatures while maintaining the traditional computation for the update gate and candidate activation. This enhances the model's ability to reset its memory based on more sophisticated temporal features.

## Backend Compatibility

The package is compatible with all Keras 3.0 backends:
- TensorFlow 2.x
- JAX
- PyTorch

However, for optimal performance, we recommend using JAX as the backend due to its efficient handling of the signature computations.

Note: While PyTorch backend is supported, JIT compilation is currently not available with PyTorch.

## Example Usage

Here's a more complete example showing how to use SignatureLSTM for time series prediction:

```python
import keras
from sig_rnn import SignatureLSTM

# Create a model for time series prediction
model = keras.Sequential([
    keras.layers.Input(shape=(100, 20)),  # 100 timesteps, 20 features
    SignatureLSTM(
        units=64,
        signature_depth=2,
        signature_input_size=5,
        return_sequences=True
    ),
    SignatureLSTM(
        units=32,
        signature_depth=2,
        signature_input_size=5,
        return_sequences=False
    ),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Single step prediction
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
```

## Implementation Details

The implementation uses the streaming signature computation from keras_sig, which allows for efficient processing of sequential data. The signature values are normalized by time to ensure stable training dynamics. Both layers support return_sequences and return_state options, making them compatible with standard Keras RNN patterns.

## Citation

If you use this package in your research, please cite our work:
```
[Citation information to be added upon paper release]
```


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
