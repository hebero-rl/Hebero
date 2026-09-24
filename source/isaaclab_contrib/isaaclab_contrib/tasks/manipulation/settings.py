# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every environment variable this benchmark reads, and how to read it.

Values are still read at the point of use, not captured at import: several
knobs are consulted while building env configs, and a launcher that exports a
variable after this module is first imported must still be honoured.

Two boolean readers, not one
----------------------------

The codebase uses two opposite conventions and they disagree on malformed
input, so collapsing them into a single ``flag()`` would silently change
behaviour:

* :func:`truthy` — true only for an explicit ``1/true/yes/on``. An unset or
  misspelled value is **false**. Use for opt-in switches.
* :func:`not_falsy` — true unless explicitly ``0/false/no/off``. A misspelled
  value is **true**. Use for on-by-default switches, where the intent is "only
  an explicit off turns this off".

``GRIPPER_CURRICULUM`` is on by default yet uses :func:`truthy` with a ``"1"``
default, so a misspelled value disables it — deliberately kept, since that is
what the code did before this module existed.

One deliberate behaviour change: empty means unset
--------------------------------------------------

:func:`number` and :func:`integer` treat an empty value as unset and return the
default. The expressions they replaced were ``float(os.getenv(NAME, "0.2"))``,
which raises ``ValueError`` on ``export NAME=`` — a plausible way to clear a
knob in a launcher script, and a confusing crash during env construction. The
string and int-or-None readers already behaved this way, so this also makes the
readers consistent with each other. It is the only respect in which these
functions differ from the code they replaced.

The knobs
---------

Data locations
    ``LIBERO_ASSETS_DATA_DIR``      LIBERO USD assets root.
    ``LIBERO_CONFIG_DIR``           Suite JSON dir; falls back next to the assets.
    ``LIBERO_ASSEMBLED_DATASET_DIR``  Assembled demo HDF5 dir.
    ``LIBERO_DAPG_DEMO_DIR``        Demo HDF5 dir for the DAPG loss.

Benchmark behaviour
    ``LIBERO_REWARD_MODE``          goal | world_state_tracking_reward | metaworld_dense.
    ``LIBERO_PRIVILEGED_CRITIC``    Asymmetric critic; on unless explicitly off.
    ``GRIPPER_CURRICULUM``          SR-gated demo replay of the gripper bit.
    ``GRIPPER_CURRICULUM_SR_THRESHOLD``  SR at which the curriculum releases (0.2).
    ``LIBERO_BOOTSTRAP_ON_TIMEOUT`` Publish ``time_outs`` so the value target bootstraps (off).
    ``LIBERO_SUCCESS_REQUIRE_TRANSITION``  Require a goal-state transition for success (on).
    ``LIBERO_SUCCESS_TERMINATION``  End a *training* episode on success too (off).
    ``LIBERO_EVALUATION``           Eval mode; disables train-only randomisation.

Demo loading
    ``LIBERO_COMPAT_REQUIRE_DEMOS`` Raise instead of falling back to zero diffs.
    ``LIBERO_COMPAT_MAX_DEMOS_PER_TASK``  Cap episodes loaded per task.
    ``LIBERO_RANDOM_INIT_STATE``    Redraw object reset XY from the demos' range (off).
    ``LIBERO_RANDOM_INIT_XY_SCALE``  Grow that range about its centre (1.0 = as demoed).
    ``LIBERO_RANDOM_INIT_TIMESTEP``  Reset to a random timestep along the demo (off).

Scene / robot
    ``ROBOT_INIT_NOISE_STD``        Gaussian noise on reset arm joints [rad] (0.0).

Metrics
    ``MTB_TASK_SR_EMA_ALPHA``       Per-task SR EMA alpha (0.05).
    ``MTB_TASK_SR_MIN_EPISODES``    Episodes before a task's SR is reported (20).
"""

from __future__ import annotations

import os

#: Values that :func:`truthy` accepts as true.
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: Values that :func:`not_falsy` accepts as false.
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def truthy(name: str, default: str = "") -> bool:
    """True only when ``name`` is set to an explicit ``1/true/yes/on``.

    Anything else — unset, empty, or misspelled — is false. Use for opt-in
    switches, and for on-by-default switches that should fail closed on a
    malformed value (pass ``default="1"``).
    """
    return os.getenv(name, default).strip().lower() in TRUE_VALUES


def not_falsy(name: str, default: str = "1") -> bool:
    """True unless ``name`` is set to an explicit ``0/false/no/off``.

    A misspelled value reads as true. Use for on-by-default switches where only
    an explicit off should disable the feature.
    """
    return os.getenv(name, default).strip().lower() not in FALSE_VALUES


def text(name: str, default: str = "") -> str:
    """Stripped string value, or ``default``."""
    return os.getenv(name, default).strip()


def number(name: str, default: float) -> float:
    """Float value, or ``default`` when unset or empty."""
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else float(default)


def integer(name: str, default: int | None = None) -> int | None:
    """Int value, or ``default`` when unset or empty."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default
