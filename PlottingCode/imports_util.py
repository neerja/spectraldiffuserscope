import jax.numpy as jnp
intial_jax = jnp.zeros((1,1))
from jax import lax
from jax import random
from flax import linen as nn # needed to install
import jax

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import os
from PIL import Image
from scipy import optimize
import scipy.signal as sp
import wandb  # needed to install
from ipywidgets import IntProgress # needed to install
from IPython.display import display
import sys
import time
import optax
import torch.nn.functional as F