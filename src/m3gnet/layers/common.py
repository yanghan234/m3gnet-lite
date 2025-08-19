import torch
from torch import nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ReLULayer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class SigmoidLayer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


class SwishLayer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x * self.sigmoid(x)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        output_dim: int | list[int],
        activation: str | list[str] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.activation = activation

        if isinstance(output_dim, int):
            output_dim = [output_dim]

        if isinstance(activation, str):
            activation = [activation] * len(output_dim)
        elif isinstance(activation, list) and len(activation) == len(output_dim):
            pass
        else:
            raise ValueError("Invalid activation type or length")

        layers = []
        _in_dim = in_dim
        for out_dim, act in zip(output_dim, activation, strict=False):
            if act is None:
                layers.append(LinearLayer(in_dim=_in_dim, out_dim=out_dim, bias=bias))
            elif act == "relu":
                layers.append(ReLULayer(in_dim=_in_dim, out_dim=out_dim, bias=bias))
            elif act == "sigmoid":
                layers.append(SigmoidLayer(in_dim=_in_dim, out_dim=out_dim, bias=bias))
            elif act == "swish":
                layers.append(SwishLayer(in_dim=_in_dim, out_dim=out_dim, bias=bias))
            else:
                raise ValueError(f"Unsupported activation function: {act}")
            _in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        output_dim: int | list[int],
        activation: str | list[str] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.mlp = MLP(
            in_dim=in_dim, output_dim=output_dim, activation=activation, bias=bias
        )
        if activation is None or isinstance(activation, str):
            activation = None
        elif isinstance(activation, list):
            activation[-1] = "sigmoid"
        self.gate = MLP(
            in_dim=in_dim, output_dim=output_dim, activation=activation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.gate(x)
