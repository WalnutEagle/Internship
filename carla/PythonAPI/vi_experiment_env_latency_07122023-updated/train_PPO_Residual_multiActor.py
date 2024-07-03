"""
train_PPO_Residual_multiActor.py
"""

import supersuit as ss
import os
import carla_env_final

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np
import torch
import pandas as pd


# ==============================================================================
# -- Callbacks----------------------------------------------------------------
# ==============================================================================

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class CustomValueCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomValueCallback, self).__init__(verbose)
        self.values_data = []  # To store values after each episode
        self.count=0

    def _on_step(self) -> bool:
        #print(dir(self.model.env))
        self.count+=1
        #if(self.count%1500==0):
            #obs = self.model.env.step(self.model.env.actions)
            #values = self.model.policy.predict_values(torch.Tensor(obs).to('cuda:1'))
        #print(list(self.model.env.buf_obs.values())[-1])
        x=list(self.model.env.buf_obs.values())[-1]
        values = self.model.policy.predict_values(torch.Tensor(x).to('cuda:1'))
            #action, value= self.model.predict(obs)
            #action, value= self.model.predict(obs)
            
            # Append the estimated value to the data list
        self.values_data.append(values.item())
            # Append the values to the data list
            #self.values_data.append(values.item())
        # print("Hello")
        values_df = pd.DataFrame({"Values": self.values_data})
        values_df.to_csv("estimated_values.csv", index=False)


def train():
    log_dir = "logs/tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env = carla_env_final.multiActorEnv()
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # env = ss.stable_baselines3_vec_env_v0(env, 1)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
    # env.reset()

    
    
    # env = Monitor(env, log_dir)
    

    # policy_kwargs = dict(net_arch=dict(pi=[16,16], vf=[256,256]))
    model = PPO("MlpPolicy", env, verbose=1, device=0)
    callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)
    value_callback = CustomValueCallback()

    # callback=CallbackList([callback, value_callback])
    model.learn(total_timesteps=15000, callback=CallbackList([callback]))

    env.close()

if __name__ == "__main__":
    train()