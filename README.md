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
The optimizer is the ADAM optimizer with a learning rate scheduler which decreases it 10% after each epoch.
The batch size is set to 50.

A couple of techniques such as early stopping and gradient clipping are in place to avoid overfitting.

## Evaluation

I'm using a 5-fold cross-validatio to evaluate the model performance.
The metrics of interest are:

- Accuracy
- Precision
- Recall
- F1 Score
- Area under the ROC (AUROC)

By default it was trained using `torch.float32` and yielded these averaged metrics across
the training folds.

```text
Class 0
---
Accuracy: 0.8803408741950989
Precision: 0.8762761950492859
Recall: 0.8803408741950989
F1: 0.8781313896179199
AUROC: 0.9888917803764343

Class 1
---
Accuracy: 0.9190875887870789
Precision: 0.9104253649711609
Recall: 0.9190875887870789
F1: 0.9146085977554321
AUROC: 0.9921445846557617

Class 2
---
Accuracy: 0.8765876889228821
Precision: 0.8842183351516724
Recall: 0.8765876889228821
F1: 0.8802887201309204
AUROC: 0.9919253587722778

Class 3
---
Accuracy: 0.9528071284294128
Precision: 0.9565860033035278
Recall: 0.9528071284294128
F1: 0.9546396136283875
AUROC: 0.992216885089874
```

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
