[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Getting Started

To get started with this project, first you should perform the setup steps in the [Udacity Deep Reinforcement Learning Nanodegree Program GitHub repository](https://github.com/udacity/deep-reinforcement-learning). Namely, you should

1. Install [Conda](https://docs.conda.io/en/latest/) and create a Python 3.6 virtual environment
2. Install [OpenAI Gym](https://github.com/openai/gym)
3. Clone the [Udacity repo]((https://github.com/udacity/deep-reinforcement-learning)) and install the Python requirements included
4. Download the Banana World Unity files appropriate for your operating system and architecture ([Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip), [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip), [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip))

Once you have performed this setup, you should be ready to run the [`Navigation.ipynb`](Navigation.ipynb) Jupyter Notebook in this repo. This notebook contains all the steps needed to define and train a DQN Agent to solve this environment.