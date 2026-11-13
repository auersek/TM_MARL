import os
import tarfile
from pathlib import Path

from torch.optim import Adam
import numpy as np
import torch

from tmrl.config import config_constants as cfg
from tmrl.util import dump, load
import logging


def load_run_instance_images_dataset(checkpoint_path):
    """
    function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" load: tar_path :{tar_path}")
    logging.debug(f" load: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'r') as t:
        t.extractall(dataset_path)
    return load(checkpoint_path)


def dump_run_instance_images_dataset(run_instance, checkpoint_path):
    """
    function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" dump: tar_path :{tar_path}")
    logging.debug(f" dump: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'w') as tar_handle:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                tar_handle.add(os.path.join(root, file), arcname=file)
    dump(run_instance, checkpoint_path)


def update_memory(run_instance):
    steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
    memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
    batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]
    if run_instance.steps != steps \
            or run_instance.memory.batch_size != batch_size \
            or run_instance.memory.memory_size != memory_size:
        run_instance.steps = steps
        run_instance.memory.nb_steps = steps
        run_instance.memory.batch_size = batch_size
        run_instance.memory.memory_size = memory_size
        logging.info(f"Memory updated with steps:{steps}, batch size:{batch_size}, memory size:{memory_size}.")
    return run_instance


def optimizer_hparams(optimizer):
    """
    Returns the hyperparameters of an optimizer that can be set from config.json, for comparison.
    """
    group = optimizer.param_groups[0]
    return dict(optimizer=type(optimizer).__name__, lr=group["lr"], betas=tuple(group.get("betas", ())), weight_decay=group.get("weight_decay"))


def update_run_instance(run_instance, training_cls):
    """
    Updates the checkpoint after loading with compatible values from config.json

    Args:
        run_instance: the instance of the checkpoint to update
        training_cls: partially instantiated class of a new checkpoint (to replace run_instance if needed)

    Returns:
        run_instance: the updated checkpoint
    """
    # the replay memory stores LIDARs as scaled when they were collected; never mix two scales in one memory:
    if cfg.PRAGMA_LIDAR:
        stored_divisor = getattr(run_instance.memory, "lidar_input_divisor", 1.0)  # older checkpoints stored raw LIDARs
        if stored_divisor != cfg.LIDAR_INPUT_DIVISOR:
            raise RuntimeError(f"The replay memory of this checkpoint holds LIDARs divided by {stored_divisor}, but "
                               f"LIDAR_INPUT_DIVISOR is {cfg.LIDAR_INPUT_DIVISOR} in config.json. Mixing both scales would "
                               f"corrupt training: set LIDAR_INPUT_DIVISOR back to {stored_divisor}, or use a new RUN_NAME.")

    # check whether we should start a new experiment entirely and keep only the memory:
    if "RESET_TRAINING" in cfg.TMRL_CONFIG and cfg.TMRL_CONFIG["RESET_TRAINING"]:
        new_run_instance = training_cls()
        new_run_instance.memory = run_instance.memory
        new_run_instance = update_memory(new_run_instance)
        new_run_instance.total_samples = len(new_run_instance.memory)
        return new_run_instance

    # the network architecture is pickled with the checkpoint, so CRITIC_LAYER_NORM cannot change on an existing run:
    if cfg.PRAGMA_LIDAR and not cfg.PRAGMA_RNN:
        model = run_instance.agent.model
        critic = model.q1 if hasattr(model, "q1") else model.qs[0]
        has_layer_norm = any(isinstance(module, torch.nn.LayerNorm) for module in critic.modules())
        wants_layer_norm = bool(cfg.TMRL_CONFIG["ALG"].get("CRITIC_LAYER_NORM", False))
        if has_layer_norm != wants_layer_norm:
            raise RuntimeError(f"The critic of this checkpoint was built {'with' if has_layer_norm else 'without'} LayerNorm, but "
                               f"CRITIC_LAYER_NORM is {wants_layer_norm} in config.json. A resumed run keeps its pickled "
                               f"architecture: set CRITIC_LAYER_NORM back to {has_layer_norm}, or use a new RUN_NAME.")

    # update training Agent:
    ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
    ALG_NAME = ALG_CONFIG["ALGORITHM"]
    assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not supported by this checkpoint updater."

    if ALG_NAME in ["SAC", "REDQSAC"]:
        lr_actor = ALG_CONFIG["LR_ACTOR"]
        lr_critic = ALG_CONFIG["LR_CRITIC"]
        lr_entropy = ALG_CONFIG["LR_ENTROPY"]
        gamma = ALG_CONFIG["GAMMA"]
        polyak = ALG_CONFIG["POLYAK"]
        learn_entropy_coef = ALG_CONFIG["LEARN_ENTROPY_COEF"]
        target_entropy = ALG_CONFIG["TARGET_ENTROPY"]
        alpha = ALG_CONFIG["ALPHA"]

        if ALG_NAME == "SAC":
            # Same config lookups as the SAC AGENT partial in config_objects.py:
            agent = run_instance.agent
            agent.lr_actor = lr_actor
            agent.lr_critic = lr_critic
            agent.optimizer_actor = ALG_CONFIG["OPTIMIZER_ACTOR"].lower()
            agent.optimizer_critic = ALG_CONFIG["OPTIMIZER_CRITIC"].lower()
            agent.betas_actor = ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None
            agent.betas_critic = ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None
            agent.l2_actor = ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None
            agent.l2_critic = ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None

            # Build the optimizers config.json asks for, and swap them in only if they differ from the checkpointed ones
            # (comparing the actual optimizer settings, so that the optimizer state is kept whenever nothing changed):
            for role, attr, make_optimizer in (("Actor", "pi_optimizer", agent.make_actor_optimizer),
                                               ("Critic", "q_optimizer", agent.make_critic_optimizer)):
                old_hparams = optimizer_hparams(getattr(agent, attr))
                new_optimizer = make_optimizer()
                new_hparams = optimizer_hparams(new_optimizer)
                if new_hparams != old_hparams:
                    setattr(agent, attr, new_optimizer)
                    logging.info(f"{role} optimizer reinitialized with {new_hparams} (old: {old_hparams}).")

        switched_entropy_learning = run_instance.agent.learn_entropy_coef != learn_entropy_coef
        if switched_entropy_learning:
            # Switch between fixed alpha (SAC v1) and learned alpha (SAC v2) on an existing checkpoint.
            # The entropy coefficient (re)starts from ALPHA in config.json; the block below creates the
            # optimizer (learned alpha) or the constant tensor (fixed alpha) that the agent needs.
            run_instance.agent.learn_entropy_coef = learn_entropy_coef
            logging.info(f"Entropy learning switched to {learn_entropy_coef} (alpha starts from {alpha}).")

        if switched_entropy_learning or run_instance.agent.lr_entropy != lr_entropy or run_instance.agent.alpha != alpha:
            run_instance.agent.lr_entropy = lr_entropy
            run_instance.agent.alpha = alpha
            device = run_instance.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if run_instance.agent.learn_entropy_coef:
                run_instance.agent.log_alpha = torch.log(torch.ones(1) * run_instance.agent.alpha).to(device).requires_grad_(True)
                run_instance.agent.alpha_optimizer = Adam([run_instance.agent.log_alpha], lr=lr_entropy)
                logging.info(f"Entropy optimizer reinitialized.")
            else:
                run_instance.agent.alpha_t = torch.tensor(float(run_instance.agent.alpha)).to(device)
                logging.info(f"Alpha changed to {alpha}.")
        
        if run_instance.agent.gamma != gamma:
            old = run_instance.agent.gamma
            run_instance.agent.gamma = gamma
            logging.info(f"Gamma coefficient changed to {gamma} (old: {old}).")

        if run_instance.agent.polyak != polyak:
            old = run_instance.agent.polyak
            run_instance.agent.polyak = polyak
            logging.info(f"Polyak coefficient changed to {polyak} (old: {old}).")

        if target_entropy is None:  # automatic entropy coefficient
            action_space = run_instance.agent.action_space
            run_instance.agent.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            run_instance.agent.target_entropy = float(target_entropy)
        logging.info(f"Target entropy: {run_instance.agent.target_entropy}.")

        if ALG_NAME == "REDQSAC":
            m = ALG_CONFIG["REDQ_M"]
            q_updates_per_policy_update = ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]

            if run_instance.agent.q_updates_per_policy_update != q_updates_per_policy_update:
                old = run_instance.agent.q_updates_per_policy_update
                run_instance.agent.q_updates_per_policy_update = q_updates_per_policy_update
                logging.info(f"Q update ratio switched to {q_updates_per_policy_update} (old: {old}).")

            if run_instance.agent.m != m:
                old = run_instance.agent.m
                run_instance.agent.m = m
                logging.info(f"M switched to {m} (old: {old}).")

    epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
    rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
    update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
    update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
    max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    profiling = cfg.PROFILE_TRAINER
    start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

    if run_instance.epochs != epochs:
        old = run_instance.epochs
        run_instance.epochs = epochs
        logging.info(f"Max epochs changed to {epochs} (old: {old}).")

    if run_instance.rounds != rounds:
        old = run_instance.rounds
        run_instance.rounds = rounds
        logging.info(f"Rounds per epoch changed to {rounds} (old: {old}).")

    if run_instance.update_model_interval != update_model_interval:
        old = run_instance.update_model_interval
        run_instance.update_model_interval = update_model_interval
        logging.info(f"Model update interval changed to {update_model_interval} (old: {old}).")

    if run_instance.update_buffer_interval != update_buffer_interval:
        old = run_instance.update_buffer_interval
        run_instance.update_buffer_interval = update_buffer_interval
        logging.info(f"Buffer update interval changed to {update_buffer_interval} (old: {old}).")

    if run_instance.max_training_steps_per_env_step != max_training_steps_per_env_step:
        old = run_instance.max_training_steps_per_env_step
        run_instance.max_training_steps_per_env_step = max_training_steps_per_env_step
        logging.info(f"Max train/env step ratio changed to {max_training_steps_per_env_step} (old: {old}).")

    if run_instance.profiling != profiling:
        old = run_instance.profiling
        run_instance.profiling = profiling
        logging.info(f"Profiling witched to {profiling} (old: {old}).")

    if run_instance.start_training != start_training:
        old = run_instance.start_training
        run_instance.start_training = start_training
        logging.info(f"Number of environment steps before training changed to {start_training} (old: {old}).")

    run_instance = update_memory(run_instance)

    return run_instance
