# Comparison between different MCTS optimizations and implementations

## Description

This project contains multiple implementations of MCTS with and without neural networks.
- MCTS
- MCTS with normalized reward
- MCTS with a ANN for improving the simulation stage
- MCTS with a AlphaZero implementation

The aim of this project was to see how good the different MCTS implementations would get at connect four. In the end, MCTS with neural nets did not perform better than normal MCTS, but with better training and some small optimizations, this could change.

## Table of Contents (Optional)

If your README is long, add a table of contents to make it easy for users to find what they need.

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Installation

Clone the project:

SSH:
```sh
git clone git@github.com:BrageHK/IDATT2502-prosjekt.git
```
HTTPS:
```sh
git clone https://github.com/BrageHK/IDATT2502-prosjekt.git
```

This project requires the following python packages
- plyer
- numpy
- torch
- torchvision
- tqdm
- windows-curses (only for windows users)

All pacckages can be installed with:
```sh
pip3 install -r requirements.txt
```
or
```sh
pip3 install -r requirements-windows.txt
```

## Usage

With this application, you can train ai, benchmark differnt MCTS and play against differnt MCTS.

### Play against ai:

```sh
python3 src/Play_vs_ai.py
```

### Train ai:
```sh
python3 src/Training_general.py
```

### Benchmark ai:
```sh
python3 src/ai_benchmark.py
```

In each of the files above, change parameters to your preference in the main function.

## Credits

<div style="text-align: center;">
  <div style="display: inline-block; text-align: center;">
    <a href="https://github.com/BrageHK">
      <img src="https://github.com/BrageHK.png" width="100" height="100" style="border-radius:50%;">
      <br>Brage Halvorsen Kvamme
    </a>
  </div>
  <div style="display: inline-block; text-align: center; margin-left: 20px;">
    <a href="https://github.com/Ewh0221">
      <img src="https://github.com/Ewh0221.png" width="100" height="100" style="border-radius:50%;">
      <br>Eilert Werner Hansen
    </a>
  </div>
  <div style="display: inline-block; text-align: center; margin-left: 20px;">
    <a href="https://github.com/HansMagneAsheim">
      <img src="https://github.com/HansMagneAsheim.png" width="100" height="100" style="border-radius:50%;">
      <br>Hans Magne Aasheim
    </a>
  </div>
</div>



## Features

- Play vs. AI in Connect Four or Tic Tac Toe
- Train your own AI
- Benchmark and get matchup statistics
- Visualize data