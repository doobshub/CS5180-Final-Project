# Mario DQN Exploration Study

This project implements and compares four DQN exploration strategies on the Super Mario Bros environment:

- **ε-greedy**: explores with a linearly decaying epsilon schedule.
- **Softmax**: samples actions from a temperature-controlled softmax distribution.
- **Heuristic**: uses a fixed heuristic action distribution when exploring.
- **Noisy Networks**: uses factorized Gaussian noise in the network's fully connected layers.

The experiment runs each strategy with three fixed random seeds (`1234`, `5678`, and `9012`). The notebook saves training returns and losses, plots learning curves, and compares the average performance of the strategies.

## Project structure

```text
.
├── project.ipynb
├── data/
│   ├── mario_dqn_egreedy_1234.pt
│   ├── mario_dqn_egreedy_1234_returns.npy
│   ├── mario_dqn_egreedy_1234_loss.npy
│   ├── ...
└── README.md
```

Each model prefix represents a strategy and seed. For example, `mario_dqn_softmax_5678` contains the best checkpoint, discounted-return history, and loss history for the softmax experiment with seed `5678`.

## Requirements

The notebook requires Python and the following packages:

- `numpy`
- `torch`
- `gym`
- `gym_super_mario_bros`
- `nes_py`
- `matplotlib`
- `tqdm`

Install them in the active Python environment with:

```bash
python -m pip install numpy torch gym gym_super_mario_bros nes_py matplotlib tqdm
```

The project also requires a CUDA-capable GPU because the DQN agent selects `torch.device("cuda")` in the notebook. If CUDA is unavailable, update the device selection to `torch.device("cpu")` before running the notebook.

## Run the experiment

1. Open `project.ipynb` in VS Code or Jupyter Notebook.
2. Select the Python environment that has the required packages installed.
3. Run the notebook cells in order.

The notebook executes the four experiments independently. Running the complete notebook trains all 12 model configurations and may take several hours on a GPU. The saved artifacts in `data/` can be used to plot results without retraining the models.

To reproduce one configuration, find the corresponding `if __name__ == '__main__':` cell and run it. The cell creates the environment, sets the fixed seed, and trains the agent using the configuration embedded in that cell.

## Outputs

For every experiment, the notebook writes:

- `<model_name>.pt`: the behavior-network state dictionary from the highest-return episode.
- `<model_name>_returns.npy`: discounted returns recorded at episode completion.
- `<model_name>_loss.npy`: training losses recorded during behavior-policy updates.

The notebook also displays individual learning curves, strategy-specific comparison plots, and a combined comparison plot. The final cells calculate mean returns for each strategy across the three seeds.

## Configuration

The shared training configuration is:

- Environment: `SuperMarioBros-v0`
- Observation: four RGB frames resized to `84 x 84`
- Discount factor: `0.99`
- Training steps: `2,000,000`
- Replay buffer: `100,000` transitions
- Batch size: `64`
- Learning rate: `0.0001`
- Target-network update interval: `2,000` steps
- Behavior-network update interval: `4` steps

The exploration variants differ in their action-selection behavior. The noisy-network variant uses a fixed sigma value of `0.1` for its Gaussian noise parameters.

## Notes

- The notebook uses a fixed seed for each run, making the saved results reproducible when the same environment and dependency versions are used.
- The model checkpoints are saved only when an episode return exceeds the best return observed so far.
- The combined plot truncates each strategy's history to the shortest available run before plotting.
- The notebook currently contains a typo in the noisy-network class name (`Nosiy_Linear`), but it remains valid Python and is used by the network implementation.
