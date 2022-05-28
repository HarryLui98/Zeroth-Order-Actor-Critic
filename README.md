<style>
table th:first-of-type {
    width: 10%;
}
table th:nth-of-type(2) {
    width: 30%;
}
table th:nth-of-type(3) {
    width: 30%;
}
</style>

## Zeroth-Order Actor-Critic (ZOAC)
#### Codes
##### ZOAC
The training code of ZOAC will be realesed in this repo.
##### Baselines
* Evolution strategies (ES) implemented in RLlib ([repo](https://github.com/ray-project/ray/tree/master/rllib))
* Augmented random search (ARS) implemented by authors of the original paper ([repo](https://github.com/modestyachts/ARS))
* Proximal policy optimization (PPO) implemented in stable-baseline3 ([repo](https://github.com/DLR-RM/stable-baselines3))
#### Additional Results on Robustness
* We compare the robustness of the linear policies learned by ARS/ZOAC. Although they have identical policy structure, ZOAC is able to find policies that are much more robust to noise.

| Algo | No Noise | Observation Noise $(\sigma=0.1)$| Parameter Noise $(\sigma=0.08)$ |
|:---:|:---:|:---:|:---:|
| ARS | ![](./figure/robust/ars_400.gif)  | ![](./figure/robust/ars_400_obs0.1.gif) | ![](./figure/robust/ars_300_para0.08.gif) |
| ZOAC | ![](./figure/robust/zoacmat_300.gif)  | ![](./figure/robust/zoacmat_300_obs0.1.gif) | ![](./figure/robust/zoacmat_300_para0.08.gif) |