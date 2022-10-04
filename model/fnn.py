from typing import List, Optional
import torch
from torch import nn

from utils.const import DEFAULT_DTYPE, DEFAULT_DEVICE


class FNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Optional[List[int]] = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device = DEFAULT_DEVICE
    ) -> None:
        """Standard Feedforward Neural Network (FNN).

        :param input_size: Input size of input.
        :type input_size: int
        :param output_size: Output size of output.
        :type output_size: int
        :param hidden_sizes: List of sizes for hidden layers, defaults to [].
        No hidden layers if empty or None.
        :type hidden_sizes: List[int], optional
        :param dtype: dtype of the model, defaults to DEFAULT_DTYPE
        :type dtype: torch.dtype, optional
        :param device: device of the model, defaults to DEFAULT_DEVICE
        :type device: torch.device, optional
        """        
        super(FNN, self).__init__()
        
        if (hidden_sizes is None) or (len(hidden_sizes) == 0):
            # no hidden layers
            self.fnn = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()
            )
        else:
            # hidden layers
            input = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU()
            )
            hidden = nn.ModuleList()
            for i in range(len(hidden_sizes) - 1):
                hidden.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                hidden.append(nn.ReLU())
            output = nn.Linear(hidden_sizes[-1], output_size)
            self.fnn = nn.Sequential(*input, *hidden, output)

        self.device = device
        self.dtype = dtype
        self.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype, device=self.device)
        if len(x.shape) == 2:
            pass
        elif len(x.shape) == 3:
            # (batch_size, num_particles, 4)
            x = x.view(x.shape[0], -1)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}.")

        return self.fnn(x)
