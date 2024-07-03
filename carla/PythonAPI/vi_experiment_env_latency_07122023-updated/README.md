# Interactive Experiment Scripts

The folder contains scripts for running the interactive experiment.

## Requirements

### Carla
- Carla 9.10.1

### Python
- Python >= 3.7
- pygame


## Running Interactive Experiment

### Installing Carla
1. Download Carla 9.10.1 distribution from: https://github.com/carla-simulator/carla/releases
2. Download AdditionalMaps_0.9.10.1
3. Unzip Carla and AdditionalMaps to `[CARLA_ROOT]`
4. Download `vi_experiment_env` to `[CARLA_ROOT]`

### Running Carla

To run Carla Server

```
bash [CARLA_ROOT]/CarlaUE4.sh 
```

### Running Scripts

First change the CARLA_ROOT variable in `[CARLA_ROOT]\vi_experiment_env\env_setup`.

Then run the experiment by

```
source [CARLA_ROOT]\vi_experiment_env\env_setup

python [CARLA_ROOT]\vi_experiment_env\run_experiment.py --filter walker

or

bash [CARLA_ROOT]\vi_experiment_env\run_experiment.sh

```

The script runs by running the `game_loop()` function which constructs and repeatedly calls two classes: 1. `World`, 2. `KeyboardControl`

To add code that runs 
- after `World` initialization, consider adding under `World.after_init()`
- every frame, consider adding under `World.tick()` or `KeyboardControl.parse_events()`
- before `World` being destroyed, consider adding under `World.before_destroy()`