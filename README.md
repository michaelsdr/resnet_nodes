# Code to reproduce the Neurips submission : "_Do Residual Neural Networks discretize Neural Ordinary Differential Equations?_"



## Compat

This package has been developed and tested with `python3.8`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`. To test that the installation has been successful, you can install `pytest` and run the test suite using

```
pip install pytest
pytest
```


## Reproducing the experiments/figures of the paper

### Experiment in Figure 2 - Illustration of the smoothness of the weights in the linear case

```bash
python experiments/expe_linear_weights.py
```
The plot is saved in the folder figures.
### Experiment in Figure 3 - (a) Train models with tied weights

On CIFAR:

```bash
python experiments/one_expe_CIFAR_10.py -m 'iresnetauto' --n_layers 8
```

the argument --n_layers correspond to the number of residual block per layer.

On ImageNet (needs cuda):

```bash
python experiments/one_expe_image_net.py -m 'iresnetauto' --n_layers 8
```

### Experiment in Figure 3 - (b) Failure of the adjoint method with a ResNet-101 on ImageNet (needs cuda)

```bash
python experiments/one_expe_image_net.py --use_backprop True
python experiments/one_expe_image_net.py --use_backprop False
```

### Experiment in Table 2 - Refine pretrained models by untying their weights

For CIFAR (pretrained model is available):

```bash
python experiments/one_expe_finetuning_CIFAR.py
```

For ImageNet (needs cuda and model needs to be pretrained first):

```bash
python experiments/one_expe_finetuning_imagenet.py
```

For experiments related to Figures 3 (c) and 4, see next section.

## Train our simple ResNet on CIFAR with Euler or Heun / with Adjoint Method or Backpropagation

Go to the resnet_cifar folder:

```bash
cd resnet_cifar
```

### Training

```bash
python one_expe_cifar.py --backprop 1 --heun 0 --depth 16
```

Available arguments are 1 (True) or 0 (False) for --backprop and --heun. Available arguments for --depth can be any integer. Make sure to run on all the possible combinations for --backprop and --heun in order to reproduce the experiment.

### Getting the results

```bash
python get_results.py
```

(Optionally modify the array depths in get_results.py to cover the different values for depth)

### Plot

Set the value of n_tries in plot_paper_heun.py to the amount of seeds you used. Then run 

```bash
python plot_paper_heun.py
```

### Gradients

Compare the gradients when using our Adjoint Method or not:

```bash
python compare_gradients.py --depth 32
```
The optional argument --heun can be passed in order to compare gradients for Heun. Results are saved automatically in metrics_gradient.

### Plot the relative norms of the gradients:

Plot the results 

```bash
python plot_grad_dist.py
```

## Memory savings:

To confirm that the memory requirements for the adjoint method are much smaller than for standard backpropagation, you can run from the root folder:

```bash
python experiments/memory.py
```

(This can take a few seconds.)

Results are plotted in the folder figures.


