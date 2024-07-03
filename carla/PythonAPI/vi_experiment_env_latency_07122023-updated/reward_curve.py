import numpy as np
import matplotlib.pyplot as plt
log_path = "ppo_2.zip"  # Change this to your model's log path
log_data = np.load(log_path.replace(".zip", ".monitor.csv"))

# Extract relevant data
timesteps = log_data['timesteps']
rewards = log_data['r']

# Plot the reward curve
plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards, label='Reward')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.legend()
plt.grid()
plt.show()