# The RealNVP and a simple Normalizing Flow implemented in Pytorch

The training and visualization can be found in the notebook and the models in the flow_model-file.

The three datasets that I trained the RealNVP on where the moon-dataset, a circle and multiple Gaussians all with a fixed distance from the origin. One model was trained for each dataset, and then samples from the distributions where generated.

![Alt text](/img/toy_datasets.png?raw=true "Title")


## Performance of trained networks presented as the output of each layer


![Alt text](/img/gaussians.png?raw=true "Title")

![Alt text](/img/circle.png?raw=true "Title")

![Alt text](/img/moons.png?raw=true "Title")

