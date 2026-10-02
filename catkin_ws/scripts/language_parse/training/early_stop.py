"""ms-swift 3.5 external plugin: validation-loss early stopping after one bad epoch."""

from swift.plugin import extra_callbacks
from transformers import EarlyStoppingCallback

extra_callbacks.append(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
