"""
Trainer-side safety net for long training runs.

The training checkpoint (``checkpoints/<RUN_NAME>_t.tcpt``) is overwritten at the end of every epoch, so a good
actor/critic is lost as soon as training moves past it. This module, called by the trainer:

- keeps periodic copies of the checkpoint in ``checkpoints/history/`` (the newest ``CHECKPOINT_HISTORY_KEEP`` of them),
- keeps a copy of the checkpoint with the best recent training return (``<RUN_NAME>_t_best.tcpt``, described by
  ``<RUN_NAME>_t_best.json``),
- stops the trainer once the policy has collapsed: ``COLLAPSE_STUCK_EPISODES`` consecutive training episodes ended by
  the "no progress" check at (or near) the earliest step it can fire, i.e. the car never really leaves the start.

To restore a saved checkpoint, stop the trainer and copy it over ``checkpoints/<RUN_NAME>_t.tcpt``.

Optional keys in config.json (defaults below):
    "CHECKPOINT_HISTORY_EVERY": 5     copy the checkpoint every N epochs (0 disables the periodic copies)
    "CHECKPOINT_HISTORY_KEEP": 4      number of periodic copies kept per run (the best copy is kept separately)
    "COLLAPSE_STUCK_EPISODES": 30     stop after N consecutive stuck episodes (0 disables stopping)
    "COLLAPSE_STUCK_MARGIN": 30       an episode is stuck if it is terminated at most this many steps after the
                                      earliest possible failure (MIN_STEPS + FAILURE_COUNTDOWN + 1 steps); collapsed
                                      policies sometimes creep forward and reset the failure countdown, so a margin is
                                      needed (on Lidar runs 1-3, healthy driving never gave more than 2 such episodes
                                      in a row with a margin of 30)
    "COLLAPSE_ARM_RETURN": 50.0       collapse detection only starts once the mean return of 10 consecutive training
                                      episodes has reached this value (early training is full of stuck episodes)
"""

# standard library imports
import json
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path

# local imports
import tmrl.config.config_constants as cfg


RETURN_WINDOW = 10  # number of recent training episodes averaged for the "best" checkpoint and for arming

HISTORY_EVERY = int(cfg.TMRL_CONFIG.get("CHECKPOINT_HISTORY_EVERY", 5))
HISTORY_KEEP = int(cfg.TMRL_CONFIG.get("CHECKPOINT_HISTORY_KEEP", 4))
STUCK_EPISODES = int(cfg.TMRL_CONFIG.get("COLLAPSE_STUCK_EPISODES", 30))
STUCK_MARGIN = int(cfg.TMRL_CONFIG.get("COLLAPSE_STUCK_MARGIN", 30))
ARM_RETURN = float(cfg.TMRL_CONFIG.get("COLLAPSE_ARM_RETURN", 50.0))

_REWARD_CONFIG = getattr(cfg, "REWARD_CONFIG", {}) or {}
# The reward function counts failures only after MIN_STEPS steps and terminates after FAILURE_COUNTDOWN + 1 consecutive
# steps without progress, so a car that never progresses is terminated at step MIN_STEPS + FAILURE_COUNTDOWN + 1:
EARLIEST_FAILURE_STEP = int(_REWARD_CONFIG.get("MIN_STEPS", 70)) + int(_REWARD_CONFIG.get("FAILURE_COUNTDOWN", 10)) + 1
STUCK_MAX_STEPS = EARLIEST_FAILURE_STEP + STUCK_MARGIN


class EpisodeMonitor:
    """
    Follows training episodes as their samples reach the trainer.

    Samples are the ``(act, obs, rew, terminated, truncated, info)`` tuples sent by the RolloutWorker; each episode
    starts with one reset sample (not a step) and ends with a terminated or truncated sample.
    This assumes a single RolloutWorker (samples of concurrent workers would interleave).
    The monitor is pickled with the training checkpoint, so its state survives restarts.
    """
    def __init__(self):
        self.nb_episodes = 0
        self.recent_returns = deque(maxlen=RETURN_WINDOW)
        self.stuck_streak = 0  # consecutive stuck episodes
        self.armed = False  # collapse detection starts once the run has shown it can drive
        self.collapse_detected = False  # latched as soon as the streak is reached (the trainer only checks per epoch)
        self._ret = 0.0  # return of the current episode so far
        self._steps = -1  # steps of the current episode so far (the reset sample is not a step)

    def add_samples(self, samples, stuck_max_steps=STUCK_MAX_STEPS, arm_return=ARM_RETURN, stuck_episodes=STUCK_EPISODES):
        for sample in samples:
            rew, terminated, truncated = sample[2], sample[3], sample[4]
            self._ret += float(rew)
            self._steps += 1
            if terminated or truncated:
                stuck = bool(terminated) and not truncated and self._steps <= stuck_max_steps
                self._end_episode(stuck, arm_return, stuck_episodes)

    def _end_episode(self, stuck, arm_return, stuck_episodes):
        self.nb_episodes += 1
        self.recent_returns.append(self._ret)
        self.stuck_streak = self.stuck_streak + 1 if stuck else 0
        if not self.armed and len(self.recent_returns) == RETURN_WINDOW and self.mean_recent_return() >= arm_return:
            self.armed = True
            logging.info(f" Collapse detection armed (mean return of the last {RETURN_WINDOW} episodes: {self.mean_recent_return():.1f}).")
        if stuck_episodes > 0 and self.armed and self.stuck_streak >= stuck_episodes:
            self.collapse_detected = True
        self._ret = 0.0
        self._steps = -1

    def mean_recent_return(self):
        return sum(self.recent_returns) / len(self.recent_returns) if self.recent_returns else float("nan")

    def reset_collapse_detection(self):
        """
        Called when the trainer starts: only stuck episodes observed by this trainer session can stop it
        (e.g. after resuming from a checkpoint restored from the history).
        """
        self.stuck_streak = 0
        self.collapse_detected = False


def get_monitor(run_instance):
    """
    Returns the EpisodeMonitor of a training instance, creating it for checkpoints saved before it existed.
    """
    monitor = getattr(run_instance, "episode_monitor", None)
    if monitor is None:
        monitor = EpisodeMonitor()
        run_instance.episode_monitor = monitor
    return monitor


def on_trainer_start(run_instance):
    """
    Called when the trainer has created or loaded its training instance.
    """
    get_monitor(run_instance).reset_collapse_detection()


def _copy_atomic(src, dst):
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _save_history(checkpoint_path, epoch, monitor, history_every, history_keep):
    checkpoint_path = Path(checkpoint_path)
    history_dir = checkpoint_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    # periodic copies (the timestamp avoids overwriting copies of an earlier branch after restoring an old checkpoint):
    if history_every > 0 and epoch % history_every == 0:
        dst = history_dir / f"{stem}_e{epoch:05d}_{time.strftime('%Y%m%d-%H%M%S')}{checkpoint_path.suffix}"
        _copy_atomic(checkpoint_path, dst)
        logging.info(f" Checkpoint copied to {dst}")
        copies = sorted(history_dir.glob(f"{stem}_e[0-9]*{checkpoint_path.suffix}"), key=lambda p: (p.stat().st_mtime, p.name))
        for old in copies[:max(0, len(copies) - history_keep)]:
            old.unlink()
            logging.info(f" Removed old checkpoint copy {old}")

    # best copy, judged by the mean return of the last RETURN_WINDOW training episodes:
    if len(monitor.recent_returns) == RETURN_WINDOW:
        mean_return = monitor.mean_recent_return()
        best_json = history_dir / f"{stem}_best.json"
        best_return = -float("inf")
        if best_json.exists():
            try:
                best_return = json.loads(best_json.read_text())["mean_return"]
            except (ValueError, KeyError):
                logging.warning(f" Could not read {best_json}, it will be overwritten.")
        if mean_return > best_return:
            dst = history_dir / f"{stem}_best{checkpoint_path.suffix}"
            _copy_atomic(checkpoint_path, dst)
            best_json.write_text(json.dumps({"epoch": epoch, "mean_return": mean_return, "episodes": monitor.nb_episodes,
                                             "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
            logging.info(f" New best checkpoint (mean return of the last {RETURN_WINDOW} episodes: {mean_return:.1f}, "
                         f"previous best: {best_return:.1f}) copied to {dst}")


def after_epoch(run_instance, checkpoint_path, checkpoint_saved,
                history_every=HISTORY_EVERY, history_keep=HISTORY_KEEP, stuck_episodes=STUCK_EPISODES):
    """
    Called by the trainer at the end of each epoch, after the checkpoint was (possibly) saved.

    Args:
        run_instance: the training instance
        checkpoint_path: path of the training checkpoint
        checkpoint_saved (bool): whether the checkpoint was saved at the end of this epoch

    Returns:
        bool: True if training should stop because the run has collapsed
    """
    monitor = get_monitor(run_instance)
    epoch = run_instance.epoch
    logging.info(f" Episodes: {monitor.nb_episodes}, mean return of the last {len(monitor.recent_returns)}: "
                 f"{monitor.mean_recent_return():.1f}, consecutive stuck episodes: {monitor.stuck_streak} "
                 f"(stop at {stuck_episodes}, detection {'armed' if monitor.armed else 'not armed yet'}).")

    if checkpoint_saved and checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            _save_history(checkpoint_path, epoch, monitor, history_every, history_keep)
        except OSError as e:  # e.g. disk full: keep training, the main checkpoint is still saved
            logging.warning(f" Could not save checkpoint history: {e}")

    if monitor.collapse_detected:
        logging.warning(f" RUN COLLAPSED: {stuck_episodes} consecutive training episodes were terminated within "
                        f"{STUCK_MAX_STEPS} steps without progress. Stopping the trainer. Saved checkpoints are in "
                        f"{Path(checkpoint_path).parent / 'history'} (copy one over {checkpoint_path} to resume from it).")
        return True
    return False
