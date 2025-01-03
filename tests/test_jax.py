import os
import tempfile
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

from sig_rnn.sig_lstm import SignatureLSTM
from sig_rnn.sig_gru import SignatureGRU


def generate_random_tensor(shape):
    """Generates random tensor for testing."""
    return np.random.randn(*shape)


@pytest.fixture(params=[2, 3])  # Test different signature depths
def depth(request):
    """Fixture providing different signature depths for testing."""
    return request.param


@pytest.fixture(params=[True, False])
def return_sequences(request):
    """Fixture for testing return_sequences parameter."""
    return request.param


@pytest.fixture(params=[True, False])
def return_state(request):
    """Fixture for testing return_state parameter."""
    return request.param


@pytest.fixture(params=[True, False])
def jit_compile(request):
    """Fixture for testing JIT compilation settings."""
    return request.param


@pytest.fixture(params=[SignatureGRU, SignatureLSTM])
def rnn_class(request):
    """Fixture providing both RNN layer classes for testing."""
    return request.param


def test_rnn_forward(rnn_class, depth, return_sequences, return_state):
    """Tests forward pass of SignatureGRU and SignatureLSTM.
    
    Verifies:
    1. Correct output shapes for different configurations
    2. Backend is correctly set to JAX
    3. Layer produces valid outputs
    """
    assert keras.backend.backend() == BACKEND
    
    batch_size, time_steps, features = 32, 10, 5
    units = 8
    
    layer = rnn_class(
        units=units,
        signature_depth=depth,
        return_sequences=return_sequences,
        return_state=return_state
    )

    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    outputs = layer(input_sequence)

    if return_state:
        if isinstance(layer, SignatureLSTM):
            output, final_h, final_c = outputs
            assert final_h.shape == (batch_size, units)
            assert final_c.shape == (batch_size, units)
        else:  # SignatureGRU
            output, final_h = outputs
            assert final_h.shape == (batch_size, units)
    else:
        output = outputs

    expected_shape = (batch_size, time_steps, units) if return_sequences else (batch_size, units)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"


def test_rnn_training(rnn_class, depth, jit_compile):
    """Tests RNN layers in training configuration.
    
    Verifies:
    1. Layer can be integrated into a Keras model
    2. Model successfully trains
    3. Loss is properly tracked
    4. Works with both JIT and non-JIT compilation
    """
    assert keras.backend.backend() == BACKEND
    
    batch_size, time_steps, features = 32, 10, 8
    units = 16
    output_dim = 4

    model = keras.Sequential([
        keras.layers.Input(shape=(time_steps, features)),
        rnn_class(units=units, signature_depth=depth),
        keras.layers.Dense(output_dim)
    ])

    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_target = generate_random_tensor((batch_size, output_dim))
    
    model.compile(optimizer='adam', loss='mse', jit_compile=jit_compile)
    history = model.fit(input_sequence, output_target, epochs=2, batch_size=16, verbose=0)

    assert 'loss' in history.history, "Model should train successfully"


def test_rnn_serialization(rnn_class, depth, return_sequences, return_state):
    """Tests end-to-end serialization of model containing RNN layer.
    
    Verifies:
    1. Model can be saved and loaded
    2. Predictions remain consistent after loading
    3. Loaded model can continue training
    """
    assert keras.backend.backend() == BACKEND
    
    batch_size, time_steps, features = 32, 10, 8
    units = 16
    output_dim = 4

    # Create and compile model
    inputs = Input(shape=(time_steps, features))
    rnn_layer = rnn_class(
        units=units,
        signature_depth=depth,
        return_sequences=return_sequences,
        return_state=return_state
    )
    
    if return_state:
        if isinstance(rnn_layer, SignatureLSTM):
            rnn_output, _, _ = rnn_layer(inputs)
        else:  # SignatureGRU
            rnn_output, _ = rnn_layer(inputs)
    else:
        rnn_output = rnn_layer(inputs)
    
    outputs = keras.layers.Dense(output_dim)(rnn_output)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    # Generate data
    x_train = generate_random_tensor((batch_size, time_steps, features))
    y_shape = (batch_size, time_steps, output_dim) if return_sequences else (batch_size, output_dim)
    y_train = generate_random_tensor(y_shape)

    # Train original model
    model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)
    predictions_before = model.predict(x_train, verbose=False)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'sig_rnn_model.keras')
        model.save(model_path)
        loaded_model = load_model(model_path)

    # Verify predictions
    predictions_after = loaded_model.predict(x_train, verbose=False)
    assert ops.all(ops.equal(predictions_before, predictions_after)), "Predictions should match after loading"

    # Verify continued training
    loaded_model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=False)


def test_rnn_config(rnn_class, depth, return_sequences, return_state):
    """Tests configuration serialization of RNN layers.
    
    Verifies:
    1. Layer can be converted to config
    2. Layer can be reconstructed from config
    3. Reconstructed layer maintains original parameters
    """
    assert keras.backend.backend() == BACKEND
    
    layer = rnn_class(
        units=16,
        signature_depth=depth,
        return_sequences=return_sequences,
        return_state=return_state
    )
    
    config = layer.get_config()
    reconstructed = rnn_class.from_config(config)

    assert layer.units == reconstructed.units, "Units should match"
    assert layer.signature_depth == reconstructed.signature_depth, "Signature depth should match"
    assert layer.return_sequences == reconstructed.return_sequences, "return_sequences should match"
    assert layer.return_state == reconstructed.return_state, "return_state should match"