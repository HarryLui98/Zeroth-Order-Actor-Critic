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
| ARS | ![figure/robust/ars.gif](figure/robust/ars.gif)  | ![figure/robust/ars_obs0.1.gif](figure/robust/ars_obs0.1.gif) | ![figure/robust/ars_para0.08.gif](figure/robust/ars_para0.08.gif) |
| ZOAC | ![figure/robust/zoacmat.gif](figure/robust/zoacmat.gif)  | ![figure/robust/zoacmat_obs0.1.gif](figure/robust/zoacmat_obs0.1.gif) | ![figure/robust/zoacmat_para0.08.gif](figure/robust/zoacmat_para0.08.gif) |