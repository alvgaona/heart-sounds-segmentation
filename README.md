# Heart Sounds Segmentation

This is the reimplementation of the first release in MATLAB.
This new implementation leverages Pytorch and Pytorch Lightning.
The architecture is exactly the same to compare results.
Additionally, it uses the same implementation for the Fourier Synchrosqueezed Transform (FSST)
which was compiled from MATLAB C++ generated code.

## Dataset

The dataset is exactly the same for obvious reasons.
It can be accessed through the `DavidSpringerHSS` dataset, created
to interact with Pytorch data loaders.

The raw dataset is a collection of CSV files, and in each file, you will
find the input signal and the associated labels.
Each sample has a label associated to it according to this description.

```text
1 -> Sound 1
2 -> Systolic interval
3 -> Sound 2
4 -> Diastolic interval
```

## Training

The training consists of creating three subsets: training, validation and testing.
I'm using a 5-fold cross-validatio to evaluate the model performance.
The optimizer is the ADAM optimizer with a learning rate scheduler which decreases it 10% after each epoch.

A couple of techniques such as early stopping and gradient clipping are in place to avoid overfitting.

## Evaluation


## Usage

To run the example yourself you need to install [pixi.sh](https://pixi.sh).
Then you will simply run:

```sh
pixi install
```

Once it finishes downloading the dependencies on your machine, you will be able to run the training and evaluation.

```sh
PYTHONPATH=. pixi run python main.py
```
