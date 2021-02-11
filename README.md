# RealNVP and a simple Normalizing Flow implemented in Pytorch

In this repo I have implemented two different normalizing flows in Pytorch and tested them on three different toy datasets. The two flows implemented are
* A simple affine transoform
* Real NVP [[1]](#1)

The implementation of the flows are located in [flow_models](flow_models.py) while a short presentation of the data and training is avaible in [Normalizinf Flows with Pytorch](Normalizing_Flows_with_Pytorch.ipynb).

A flow can, if its transform is invertible, be used to both learn a probability density function and sample from it. Both flows implemented in this repo are invertible, so example samples drawn from the flows are presented in the notebook. 

Bellow follows illustrations of how a 4 layer RealNVP-model transforms data sampled from a standard Mutivariate Gaussian, into the three presented distributions. A separate model was trained for each distribution. 


![Alt text](/img/toy_datasets.png?raw=true "Toy datasets")


## Performance of trained networks presented as the output of each layer

### RealNVP performance on mutiple Gaussians, located the same distance from the center

![Alt text](/img/gaussians.png?raw=true "Title")

### RealNVP performance on a circle dataset

![Alt text](/img/circle.png?raw=true "Title")

### RealNVP performance on the moon dataset

![Alt text](/img/moons.png?raw=true "Title")


## References
<a id="1">[1]</a> 
[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803), Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio, 2017.
