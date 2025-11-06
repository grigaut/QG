"""MQGeometry Fork."""

import torch

from mqgeometry.logging.core import setup_root_logger

# Set the seed for reproducibility
torch.random.manual_seed(0)
# Logging
setup_root_logger(1)
