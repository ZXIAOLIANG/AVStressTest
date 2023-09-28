# Aggresiveness-regulated Multi-agent Stress Testing of Autonomous Vehicles

This repo dedicates to the task of stress testing a black-box autnomous vehicle system using multi-agent reinforcement learnign algorithms under an adversarial environment.

The simulation environment used are included as a submodule, which is adapted from [highway-env](https://github.com/Farama-Foundation/HighwayEnv).

The codes in this repo follow the suggested one-file implementation in reinforcement learning community.

There are four algorithms experimented in this repo:
- [MACPO](https://arxiv.org/abs/2110.02793) (Multi-Agent Constrained Policy Optimization)
- [HATRPO](https://arxiv.org/abs/2109.11251) (Heterogeneous-Agent Trust Region Policy Optimization)
- [CPO](https://arxiv.org/abs/1705.10528) (Constrained Policy Optimization)
- [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization)

The results are summarized in this [thesis](http://hdl.handle.net/10012/19897)

## Environment Setup
The python environemnt can be created using conda:
```bash
conda env create -f environment.yml
```

## Running
To run the training script of an algorithm `ALGO` listed above `ALGO.py`:
```bash
python ALGO.py
```
To run the testing script of an algorithm `ALGO` listed above `ALGO_vis.py`:
```bash
python ALGO_vis.py
```

## Acknowledgments
The code within this repository was developed with assistance from [MACPO](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) and [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl).