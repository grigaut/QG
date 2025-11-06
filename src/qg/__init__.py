"""MQGeometry Fork."""

import torch
from dotenv import load_dotenv

from qg.logging.core import setup_root_logger

# Load Environment variables
load_dotenv()
# Set the seed for reproducibility
torch.random.manual_seed(0)
# Logging
setup_root_logger(1)
