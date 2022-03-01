# Notes
## 1. Reparameterization Trick
Instead of sampling by `Normal(mu, std)`, action sampling by
```
mu, std = self.forward(state)
dist = Normal(0, 1)
z = dist.sample(mu.shape).to(device)
action = torch.tanh(mu + std*z)
```
has several advantages:
- Computation reducing.
- Simple derivation.
- Numerical stability.

## 2. Enforcing Action Bounds
This trick originated from *origin paper Appendix.C* is to compute probability of `tanh(u)`.
```
log_prob = dist.log_prob(z) - torch.log(1 - sample_action.pow(2) + 1e-7)
log_prob = log_prob.sum(dim=1, keepdim=True)
```
But such implementation is numerically unstable. Please replace by
```
log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
log_prob -= (2*(np.log(2) - action - F.softplus(-2* action))).sum(axis=1, keepdim=True)
```
see https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py

## 3. Nameless Actor Update Formula
In `SAC_v1.py`, actor loss is computed by:
```
log_policy_target = excepted_new_Q - current_V
pi_loss = alpha * log_prob * (alpha * log_prob - log_policy_target).detach()
```
However, there is an alternative implementation from origin paper:
```
pi_loss = alpha * log_prob - excepted_new_Q
```
Comparing two methods, agents with the first can converge more fast, although they still converge as the iteration progresses. Unfortunately, the source of the first form is unknown.
