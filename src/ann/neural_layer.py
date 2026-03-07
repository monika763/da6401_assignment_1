"""Reusable fully connected layer for NumPy networks."""

from __future__ import annotations

import numpy as np

from .activations import get_activation


class NeuralLayer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        initialization_method: str = "xavier",
        random_seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.initialization_method = initialization_method.lower()

        self.activation, self.activation_derivative = get_activation(activation)

        rng = np.random.default_rng(random_seed)
        self.weights = self._initialize_weights(rng)
        self.bias = np.zeros((1, self.output_dim), dtype=np.float64)

        self._last_x: np.ndarray | None = None
        self._last_z: np.ndarray | None = None

    def _initialize_weights(self, rng: np.random.Generator) -> np.ndarray:
        fan_in = self.input_dim
        fan_out = self.output_dim

        if self.initialization_method == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.standard_normal((fan_in, fan_out)) * std

        if self.initialization_method == "he":
            std = np.sqrt(2.0 / fan_in)
            return rng.standard_normal((fan_in, fan_out)) * std

        if self.initialization_method == "normal":
            return rng.standard_normal((fan_in, fan_out)) * 0.01

        if self.initialization_method == "zeros":
            return np.zeros((fan_in, fan_out), dtype=np.float64)

        raise ValueError("Unsupported initialization")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_x = x
        self._last_z = x @ self.weights + self.bias
        return self.activation(self._last_z)

    def backward(self, dactivation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._last_x is None or self._last_z is None:
            raise RuntimeError("forward() must be called before backward().")

        dz = dactivation * self.activation_derivative(self._last_z)
        grad_w = self._last_x.T @ dz
        grad_b = np.sum(dz, axis=0, keepdims=True)
        dprev = dz @ self.weights.T
        return dprev, grad_w, grad_b

    def apply_gradients(self, grad_w: np.ndarray, grad_b: np.ndarray, learning_rate: float) -> None:
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b
