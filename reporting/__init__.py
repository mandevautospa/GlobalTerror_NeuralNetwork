"""reporting – utilities for saving per-run training artefacts.

Provides ``save_run_artifacts`` which writes a timestamped run folder under
``config["reporting"]["output_dir"]`` containing:

* ``config.json``           – the full config dict used for the run.
* ``git_info.json``         – current commit hash and dirty status.
* ``metrics.csv``           – per-epoch training history.
* ``plots/``                – per-head loss/accuracy curves as PNG files.
* ``confusion_matrices/``   – confusion-matrix PNGs per classification head.
"""

from .reporter import save_run_artifacts

__all__ = ["save_run_artifacts"]
