# Reinfrocement Learning Project

Final project for the Reinforcement Learnning subject. 
This project implements two reinforcement learning algorithms to train an agent to play the Atari games Breakout, Donkey Kong and Pong.

---

## Table of Contents

- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Training](#running-the-training)
  - [Evaluating the Model](#evaluating-the-model)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Demo

Here is a demonstration of the agents's performance:

### Breakout

![DQN](data/video_noisy4.gif)

![REINFORCE](data/video_noisy4.gif)

### Donkey Kong

![PPO](data/video_noisy4.gif)

![A2C](data/video_noisy4.gif)

### Pong

![PPO](data/video_noisy4.gif)


## Project Structure
Reinfrocement-Learning-Project/
├── data/                      # Folder containing videos of the agents's performance
│   ├── demo.gif               # Example GIF showing the agent's gameplay
├── src/                       # Source code and executable scripts
│   ├── train.py               # Script to train the agent
│   ├── evaluate.py            # Script to evaluate the agent
│   ├── callbacks.py           # Custom callbacks used in training
│   └── utils.py               # Utility functions for the project
├── models/                    # Folder to store saved models
│   ├── best_model.zip         # Best-performing model saved during training
│   └── checkpoint_100000.zip  # Model checkpoint at 100,000 steps
├── requirements.txt           # File specifying project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files and directories to ignore in version control


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Arroch35/Reinfrocement-Learning-Project.git
   cd Reinfrocement-Learning-Project

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## Usage

