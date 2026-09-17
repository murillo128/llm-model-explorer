"""Temporary source snapshot for the model-owned descriptor development PR.

The connected editing environment cannot clone this public repository. Capture
only tracked project source in the existing external acceptance artifact, never
credentials, model payloads, installed dependencies, or runner state. Remove this
bootstrap helper before the feature PR becomes ready for review.
"""

import os
import subprocess
import zipfile
from pathlib import Path

import pytest


 def_placeholder = None
