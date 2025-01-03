import keras
from keras import ops
from keras.layers import Layer, Dense
import numpy as np
from typing import Optional, Tuple, Union
from keras_sig import SigLayer

@keras.utils.register_keras_serializable(name="SignatureGRU")
class SignatureGRU(Layer):
    def __init__(
        self,
        units: int,
        signature_depth: int = 2,
        signature_input_size: int = 5,
        return_sequences: bool = False,
        return_state: bool = False,
        unroll_level: Union[bool,int] = 10,
        **kwargs
    ):
        """
        SignatureGRU layer that uses path signatures for reset gate only.
        Other gates use standard GRU computations.
        
        Args:
            units: Dimensionality of the output space
            signature_depth: Maximum depth for signature computation
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return states in addition to output
        """
        super().__init__(**kwargs)
        self.units = units
        self.signature_depth = signature_depth
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.signature_input_size = signature_input_size
        
        # Will be set in build()
        self.state_size = units
        self.signature_dim = None
        self.reset_kernel = None    # Kernel for signature-based reset gate
        self.input_kernel = None    # Standard kernels for update gate and candidate
        self.recurrent_kernel = None
        self.bias = None
        self.unroll_level = unroll_level 
        
    def build(self, input_shape):
        batch_size, seq_len, features = input_shape

        self.signature = SigLayer(self.signature_depth, stream=True)
        self.signature.build((batch_size, seq_len, self.signature_input_size))
        
        # Calculate signature dimension
        self.signature_dim = sum(self.signature_input_size ** i for i in range(1, self.signature_depth + 1))
        
        # Kernel for signature-based reset gate
        self.reset_kernel = self.add_weight(
            shape=(self.signature_dim, self.units),
            initializer='glorot_uniform',
            name='reset_kernel'
        )
        
        # Standard GRU kernels for update gate and candidate
        self.input_kernel = self.add_weight(
            shape=(features, self.units * 2),  # 2 for update gate and candidate
            initializer='glorot_uniform',
            name='input_kernel'
        )
        
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),  # 2 for update gate and candidate
            initializer='glorot_uniform',
            name='recurrent_kernel'
        )
        
        # Bias for all gates (including reset gate)
        self.bias = self.add_weight(
            shape=(self.units * 3,),  # 3 for update, reset, and candidate
            initializer='zeros',
            name='bias'
        )

        self.linear_preprocess_inputs_for_sig = Dense(self.signature_input_size, 'linear', use_bias=False)
        self.linear_preprocess_inputs_for_sig.build(input_shape)
        
        super().build(input_shape)
    
    def get_initial_state(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.zeros((batch_size, self.units))
    
    def _normalize_signature_by_time(self, signatures):
        """Normalize signature values by dividing by time index."""
        seq_length = signatures.shape[1]
        time_indices = ops.arange(1, seq_length + 1, dtype=signatures.dtype)
        time_indices = ops.reshape(time_indices, (1, -1, 1))
        # Add zero signature for first timestep
        return ops.concatenate([
            ops.zeros_like(signatures[:,:1,:]), 
            signatures / time_indices
        ], axis=1)

    def call(self, inputs, initial_state=None, training=None):
        if keras.backend.backend() == 'tensorflow':
            return self._call_tensorflow(inputs, initial_state=initial_state, training=training)
        else:
            return self._call_generic(inputs, initial_state=initial_state, training=training)
    

    def _call_tensorflow(self, inputs, initial_state=None, training=None):
        """Process inputs using hybrid signature/standard GRU computation.
        
        Args:
            inputs: Input tensor of shape (batch_size, time_steps, features)
            initial_state: Optional initial state
            
        Returns:
            If return_sequences=False: Final output (batch_size, units)
            If return_sequences=True: Full sequence (batch_size, time_steps, units)
            If return_state=True: Tuple of (output, final_state)
        """
        # Compute signatures for reset gate
        signatures = self.signature(self.linear_preprocess_inputs_for_sig(inputs))
        normalized_signatures = self._normalize_signature_by_time(signatures)
        
        # Get sequence length and pre-compute input transformations for all timesteps
        time_steps = ops.shape(inputs)[1]
        
        # Pre-compute all input transformations at once, maintaining batch dimension
        all_x_transformed = ops.einsum('bti,ij->btj', inputs, self.input_kernel)
        
        # Initialize state
        if initial_state is None:
            h_tm1 = self.get_initial_state(all_x_transformed)
        else:
            h_tm1 = initial_state
            
        # Initialize list for sequence outputs if needed
        sequence_outputs = [] if self.return_sequences else None
        
        # Get biases for each component
        b_z, b_r, b_n = ops.split(self.bias, 3)
        
        # Process sequence
        for t in range(time_steps):
            # Get current signature and pre-computed input transform
            current_sig = normalized_signatures[:, t]
            current_x_transformed = all_x_transformed[:, t]
            
            # Standard GRU computations for update gate and candidate
            h_tm1_transformed = ops.dot(h_tm1, self.recurrent_kernel)
            
            # Combine transformations for standard gates (update and candidate)
            gates_standard = ops.add(h_tm1_transformed, current_x_transformed)
            
            # Split standard gates
            z_t, n_t = ops.split(gates_standard, 2, axis=-1)
            
            # Compute reset gate using signature
            r_t = ops.dot(current_sig, self.reset_kernel) + b_r
            
            # Apply activations
            z_t = ops.sigmoid(z_t + b_z)  # update gate
            r_t = ops.sigmoid(r_t)        # reset gate (signature-based)
            
            # Compute candidate state
            n_t = ops.tanh(n_t + b_n + r_t * ops.dot(h_tm1, self.recurrent_kernel[:, :self.units]))
            
            # Compute new state
            h_tm1 = z_t * h_tm1 + (1 - z_t) * n_t
            
            # Store output if returning sequences
            if self.return_sequences:
                sequence_outputs.append(h_tm1)
        
        if self.return_sequences:
            final_outputs = ops.stack(sequence_outputs, axis=1)
        else:
            final_outputs = h_tm1
            
        if self.return_state:
            return final_outputs, h_tm1
        return final_outputs
    
    def _call_generic(self, inputs, initial_state=None, training=None):
        """Process inputs using hybrid signature/standard GRU computation with scan operations.
        
        Args:
            inputs: Input tensor of shape (batch_size, time_steps, features)
            initial_state: Optional initial state
            
        Returns:
            If return_sequences=False: Final output (batch_size, units)
            If return_sequences=True: Full sequence (batch_size, time_steps, units)
            If return_state=True: Tuple of (output, final_state)
        """
        # Compute signatures for reset gate
        signatures = self.signature(self.linear_preprocess_inputs_for_sig(inputs))
        normalized_signatures = self._normalize_signature_by_time(signatures)
        
        # Pre-compute input transformations for all timesteps
        all_x_transformed = ops.einsum('bti,ij->btj', inputs, self.input_kernel)
        
        # Initialize state
        if initial_state is None:
            initial_h = self.get_initial_state(all_x_transformed)
        else:
            initial_h = initial_state
        
        # Get biases for each component
        b_z, b_r, b_n = ops.split(self.bias, 3)
        
        # Define scan function for processing sequence
        def step_fn(h_prev, step_inputs):
            # Unpack current step inputs
            current_sig, current_x_transformed = step_inputs
            
            # Standard GRU computations for update gate and candidate
            h_prev_transformed = ops.dot(h_prev, self.recurrent_kernel)
            
            # Combine transformations for standard gates
            gates_standard = ops.add(h_prev_transformed, current_x_transformed)
            
            # Split standard gates
            z_t, n_t = ops.split(gates_standard, 2, axis=-1)
            
            # Compute reset gate using signature
            r_t = ops.dot(current_sig, self.reset_kernel) + b_r
            
            # Apply activations
            z_t = ops.sigmoid(z_t + b_z)  # update gate
            r_t = ops.sigmoid(r_t)        # reset gate (signature-based)
            
            # Compute candidate state
            n_t = ops.tanh(n_t + b_n + r_t * ops.dot(h_prev, self.recurrent_kernel[:, :self.units]))
            
            # Compute new state
            h_t = z_t * h_prev + (1 - z_t) * n_t
            
            if self.return_sequences:
                return h_t, h_t
            return h_t, None
        
        # Prepare inputs for scan
        scan_inputs = (
            ops.moveaxis(normalized_signatures, 1, 0),  # (time_steps, batch, sig_dim)
            ops.moveaxis(all_x_transformed, 1, 0)      # (time_steps, batch, transformed_dim)
        )
        
        # Process sequence using scan
        final_state, sequence = ops.scan(
            step_fn,
            init=initial_h,
            xs=scan_inputs,
            unroll=self.unroll_level,
            length=ops.shape(inputs)[1]
        )
        
        if self.return_sequences:
            # Restore time axis to middle position
            outputs = ops.moveaxis(sequence, 0, 1)
        else:
            outputs = final_state
            
        if self.return_state:
            return outputs, final_state
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'signature_depth': self.signature_depth,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
        })
        return config
        
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        if input_shape is None or input_shape[0] is None:
            batch_size = None
        else:
            batch_size = input_shape[0]
            
        if self.return_sequences:
            output_shape = (batch_size, input_shape[1], self.units)
        else:
            output_shape = (batch_size, self.units)
            
        if self.return_state:
            return [output_shape, (batch_size, self.units)]
        return output_shape
        