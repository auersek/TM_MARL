# third-party imports
import numpy as np
import logging

# local imports
import tmrl.config.config_constants as cfg


# OBSERVATION PREPROCESSING ==================================


def obs_preprocessor_tm_act_in_obs(obs):
    """
    Preprocessor for TM2020 full environment with grayscale images
    """
    grayscale_images = obs[3]
    grayscale_images = grayscale_images.astype(np.float32) / 256.0
    obs = (obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10000.0, grayscale_images, *obs[4:])  # >= 1 action
    return obs


def _scale_lidar(lidar):
    """
    Flattens LIDARs and divides them by LIDAR_INPUT_DIVISOR (config.json, 1.0 = raw pixel distances)
    """
    lidar = np.ndarray.flatten(lidar)
    if cfg.LIDAR_INPUT_DIVISOR != 1.0:
        lidar = lidar / np.float32(cfg.LIDAR_INPUT_DIVISOR)
    return lidar


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening (and scaling) LIDARs
    """
    obs = (obs[0], _scale_lidar(obs[1]), *obs[2:])  # >= 1  action
    return obs


def obs_preprocessor_tm_lidar_progress_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening (and scaling) LIDARs
    """
    obs = (obs[0], obs[1], _scale_lidar(obs[2]), *obs[3:])  # >= 1  action
    return obs


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC


def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated):
    return last_obs, act, rew, new_obs, terminated, truncated
