## Zeroth-Order Actor-Critic (ZOAC)
### Codes
#### ZOAC
The complete training code of ZOAC will be realesed in this repo.
At present, some of the learned policies and testing codes are provided for demonstration.
#### Baselines
* Evolution strategies (ES) implemented in RLlib ([repo](https://github.com/ray-project/ray/tree/master/rllib))
* Augmented random search (ARS) implemented by authors of the original paper ([repo](https://github.com/modestyachts/ARS))
* Proximal policy optimization (PPO) implemented in stable-baseline3 ([repo](https://github.com/DLR-RM/stable-baselines3))
### Additional Results on Robustness
* Apart from the results presented in the paper, we further visualize the linear policies learned by ARS/ZOAC below.
  Although they have identical policy structure, ZOAC is able to find policies that are much more robust to noise.

| Algo | No Noise | Observation Noise $(\sigma=0.1)$| Parameter Noise $(\sigma=0.08)$ |
|:---:|:---:|:---:|:---:|
| ARS | [ars.gif](https://i.ibb.co/02dK5jP/ars.gif)  | [ars_obs0.1.gif](https://i.ibb.co/TKZQWkp/ars-obs0-1.gif) | [ars_para0.08.gif](https://i.ibb.co/bPbKJYz/ars-para0-08.gif) |
| ZOAC | [zoacmat.gif](https://i.ibb.co/xCmyyB2/zoacmat.gif)  | [zoacmat_obs0.1.gif](https://i.ibb.co/tZbV4C5/zoacmat-obs0-1.gif) | [zoacmat_para0.08.gif](https://i.ibb.co/4FKbwmj/zoacmat-para0-08.gif) |