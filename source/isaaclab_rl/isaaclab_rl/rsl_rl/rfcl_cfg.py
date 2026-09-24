from __future__ import annotations

from dataclasses import field
from isaaclab.utils.configclass import configclass


@configclass
class RslRlRFCLCfg:
    """Configuration for Reverse Forward Curriculum Learning (RFCL) with PPO.

    Aligned with the official implementation from Tao et al. 2024
    (https://github.com/StoneT2000/rfcl).  Two-phase curriculum over
    demonstration trajectory states:

    Phase 1 — Reverse curriculum:
        Each demo τ_i has its own cursor ``t_i`` (initialised to ``T_i - 1``).
        At every reset the per-episode start step is drawn from a distribution
        ``K`` around ``t_i`` (see ``start_step_sampler``).  Only frontier
        samples (``ptr == t_i``) feed the per-demo success buffer; once the
        last ``per_demo_buffer_size`` frontier episodes ALL succeed, the
        cursor advances backward by ``reverse_step_size``.

    Phase 2 — Forward training from every demo's ``ptr=0`` state (per-task):
        Each task individually transitions to Phase 2 when its *active* Phase 1
        demo subset (the first ``phase1_demo_count`` demos, or all demos when
        that is None) has every cursor at ``min_ptr``.  Phase 2 then unlocks the
        full demo pool, draws a demo uniformly at each reset and injects that
        demo's ``ptr=0`` state — each demo's first timestep is a distinct natural
        initial condition, so Phase 2 widens the init-state distribution beyond
        the prefix the reverse curriculum trained on.

        Demo-state injection does **not** stop in Phase 2: the env's own reset
        chain (``set_libero_default_states`` / ``randomize_object_pose``) never
        takes over, so Phase 2 initial states are the demos' t=0 states, not
        randomised env defaults.

        The switch is per-task rather than the paper's single global SR
        threshold, which switched every task at once: slow tasks were dragged
        into Phase 2 by fast tasks raising the global SR, and the resulting
        distribution shift blew up the critic loss at the switch boundary.
    """

    demo_hdf5_paths: list[str] = field(default_factory=list)
    """Replayed-demo HDF5 paths ordered to match ``task_sequence`` in the env config.

    Each file must contain a ``states/`` group (or ``initial_state/`` with T > 1) holding
    per-timestep physics states (articulation joint positions/velocities + rigid-object
    poses/velocities).

    Mutually exclusive with ``demo_hdf5_dir``; if both are provided, ``demo_hdf5_paths``
    takes precedence.
    """

    demo_hdf5_dir: str | None = None
    """Directory containing replayed-demo HDF5 files (``*.hdf5``).

    When set and ``demo_hdf5_paths`` is empty, all ``*.hdf5`` files in this directory are
    discovered automatically (sorted alphabetically) and used as the ordered path list.

    The number of discovered files **must equal** the number of unique tasks in the env.
    ``RFCLEnvWrapper`` enforces this at startup and raises ``ValueError`` on a mismatch so
    misconfigurations fail loudly rather than silently skipping tasks.
    """

    task_sequence: list[str] | None = None
    """Optional ordered list of task name strings used to map each HDF5 path to
    an integer task id that matches ``task_assignment_ids`` from the env extras.
    When None, the index position in ``demo_hdf5_paths`` is used as the task id."""

    # --- Phase 1: reverse curriculum -----------------------------------------

    reverse_step_size: int = 8
    """Phase 1: fixed cursor advance step ``δ``.  Official paper default is 8."""

    per_demo_buffer_size: int = 3
    """Phase 1: ``m`` in the paper — number of last *frontier* episodes (sampled
    with ``ptr == t_i``) that must ALL succeed before the cursor advances.
    Buffer is pre-filled with zeros so a single random success cannot trigger
    advancement; ``mean(buffer) >= 1.0`` requires all ``m`` entries to be 1."""

    start_step_sampler: str = "geometric"
    """Phase 1: distribution ``K`` of start steps around the per-demo cursor
    ``t_i``.  Only samples at ``t_i`` count toward the advance criterion;
    non-frontier samples are "rehearsal" — they explore easier states
    closer to the goal to prevent catastrophic forgetting.

    Options (mirror the official impl):
      - ``"fixed_point"`` — 100% at ``t_i``.  Equivalent to the previous
        behaviour of this codebase before alignment.
      - ``"geometric"`` (default) — 5 candidates ``[t_i, t_i+1, t_i+2, t_i+3,
        t_i+4]`` with density ``[0.5, 0.25, 0.125, 0.0625, 0.0625]``.
      - ``"uniform"`` — uniform over ``[0, T_i)``.
      - ``"uniform_spike"`` — 50% at ``t_i``, 50% uniform over ``[0, T_i)``.
      - ``"uniform_step"`` — 50% at ``t_i``, 50% uniform over ``[0, t_i]``.
    """

    demo_density_mode: str = "adaptive"
    """Phase 1: probability with which each demo within a task is selected
    at reset time.

    - ``"uniform"``: equal probability ``1/n_demos``.  Pre-alignment default.
    - ``"adaptive"`` (default, matches paper): density ∝ ``t_i / T_i`` so
      un-solved demos (large ``t_i``) get more weight.  Reverse-solved demos
      (``t_i == 0``) drop to ``1e-6`` instead of zero so they are still
      rarely sampled for stability checks.
    """

    min_ptr: int = 0
    """Earliest timestep the Phase 1 reverse cursor is allowed to reach.
    Setting ``min_ptr > 0`` prevents starting from the very beginning of a demo,
    which can be useful when the first few timesteps are near-trivial approach moves."""

    phase1_demo_count: int | None = None
    """If set, Phase 1 only operates on the first ``phase1_demo_count`` demos of
    each task — cursor advance, demo sampling, and the all-cursors-at-min_ptr
    Phase 2 trigger only consider this prefix.  The remaining demos are held
    out and **only** become samplable once the task enters Phase 2.

    Use case: train the reverse curriculum on a subset of demos (e.g. 25/50)
    for sample efficiency, then expand the init-state distribution by adding
    fresh demo trajectories at Phase 2.  Each held-out demo contributes a new
    natural initial condition (``ptr=0`` state) that the Phase-1-trained policy
    must generalise to.

    None (default) → all demos used in both phases (paper-aligned behaviour)."""

    init_cursor_progress: float = 1.0
    """Initial Phase 1 cursor as a fraction of (T_i - 1).  ``1.0`` (default) starts
    the curriculum at the very end of each demo (the paper's original choice).
    Lower values (e.g. ``0.85``) skip the near-goal "easy" prefix and start the
    reverse curriculum at a slightly earlier timestep — useful when episodes near
    the goal are trivially solved and inflate frontier SR before real learning."""

    sr_ema_alpha: float = 0.05
    """EMA smoothing factor for global / per-task SR monitoring signals.
    Per-demo cursor advancement is NOT EMA-based — it uses the
    ``per_demo_buffer_size`` deque criterion."""

    # --- Phase 2: forward curriculum -----------------------------------------
    # The three fields below are retained for backward-compatibility with old
    # configs but are NO LONGER USED.  Phase 2 is now entered per-task when a
    # task's active demo cursors all reach ``min_ptr``; it then samples uniformly
    # from the full demo pool and injects each demo's ``ptr=0`` state (see
    # ``RFCLCurriculumScheduler.get_initial_state_and_ptr``).  There is no
    # prioritised demo-timestep sampling.  Object-pose randomisation on top of
    # those states still requires ``LIBERO_RANDOMIZE_OBJECT_POSE=1``.

    phase2_sr_threshold: float = 0.7
    """**Deprecated, unused.** Previously: global SR threshold that triggered a
    one-shot Phase 1 → Phase 2 switch across all tasks.  Removed because slow
    tasks were dragged into Phase 2 by fast ones and the critic could not
    handle the abrupt state-distribution shift."""

    phase2_staleness_coef: float = 0.1
    """**Deprecated, unused.** Previously: weight for staleness in Phase 2
    prioritised timestep sampling."""

    phase2_min_score: float = 0.01
    """**Deprecated, unused.** Previously: floor on Phase 2 per-timestep sampling
    probability."""
