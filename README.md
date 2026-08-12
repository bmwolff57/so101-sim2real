# so101-sim2real
Sim-to-real reinforcement learning on a low-cost SO-101 arm, using JAX and MuJoCo MJX for massively parallel training.

Training manipulation policies in simulation and deploying them on real hardware — a $300 SO-101 arm, an 8GB laptop GPU, and a JAX-first stack. Most work in this ecosystem uses PyTorch and imitation learning; this uses MuJoCo MJX for parallel rollouts and evolutionary strategies / PPO for policy search, with the goal of measuring which simulation-fidelity factors actually determine transfer on cheap hardware.

In progress. Follow along or don't.