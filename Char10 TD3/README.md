# Notes
## 1. Compared to DDPG
- **Twin Q network.** There are stills two Q networks: one for **current Q network**, the other for **target Q network**, but every Q network has two **sub-Q-network**. And when calculating the **Q value**, just choose the minimal one as the lower bound.
- **Policy Smoothing.** Adding **Gaussian Noise** to old action when updating actor and critic.
- **Delayed update.** Wrong Q network would broadcast wrong gradients to actor network, and actor would improve itself in a wrong way. Delayed update can ensure Q network to learn more and provide a more precious estimations.

## 2. Parameter Sensitivity
TD3 is simple and easy to implement, but unfortunately to make it is greatlty sensitive to hyperparameter such as learning rate, buffer size or additional noise when training. Please don't trust any default configuration.
