# Heart Sounds Segmentation

This project implements an automated heart sound segmentation system using deep learning. Originally developed in MATLAB,
this version has been reimplemented using PyTorch and PyTorch Lightning frameworks while maintaining the same neural
network architecture for direct performance comparison. The system utilizes the Fourier Synchrosqueezed Transform (FSST)
for signal processing, implemented using MATLAB-generated C++ code.

## Dataset

The project uses the DavidSpringerHSS dataset, which has been adapted for seamless integration with PyTorch data loaders.
This dataset consists of CSV files containing heart sound recordings and their corresponding segmentation labels.

The labeling scheme is as follows:

```text
1 -> Sound 1 (S1)
2 -> Systolic interval
3 -> Sound 2 (S2)
4 -> Diastolic interval
```

These labels represent the four key components of the cardiac cycle that the model aims to identify.

## Training

The training pipeline splits the data into three subsets: training, validation, and testing. The model is optimized using
the ADAM optimizer with a dynamic learning rate that decreases by 10% after each epoch. Training is performed with a batch
size of 50.

To prevent overfitting, the implementation incorporates several regularization techniques:
- Early stopping to halt training when validation performance plateaus
- Gradient clipping to stabilize training
- Learning rate scheduling for optimal convergence

## Evaluation

Model performance is rigorously evaluated using 10-fold cross-validation. The following metrics are tracked to ensure
comprehensive performance assessment:

- Accuracy: Overall correctness of predictions
- Precision: Measure of prediction quality
- Recall: Measure of prediction completeness
- F1 Score: Harmonic mean of precision and recall
- Area under the ROC (AUROC): Overall classification performance

The model was trained using `torch.float32` precision, achieving impressive results across all metrics as shown below:

| Class    | Accuracy (mean ± std)  | Precision (mean ± std) | Recall (mean ± std)  | F1 (mean ± std)    | AUROC (mean ± std)    |
|----------|------------------------|------------------------|----------------------|--------------------|-----------------------|
| Class 0  | 0.8966 ± 0.0148        | 0.8812 ± 0.0171        | 0.8966 ± 0.0148      | 0.8887 ± 0.0117    | 0.9908 ± 0.0019       |
| Class 1  | 0.9226 ± 0.0089        | 0.9252 ± 0.0136        | 0.9226 ± 0.0089      | 0.9238 ± 0.0103    | 0.9937 ± 0.0020       |
| Class 2  | 0.8891 ± 0.0141        | 0.8920 ± 0.0107        | 0.8891 ± 0.0141      | 0.8905 ± 0.0119    | 0.9934 ± 0.0017       |
| Class 3  | 0.9585 ± 0.0078        | 0.9623 ± 0.0059        | 0.9585 ± 0.0078      | 0.9604 ± 0.0055    | 0.9939 ± 0.0018       |
| Average  | 0.9167 ± 0.0114        | 0.9152 ± 0.0118        | 0.9167 ± 0.0114      | 0.9159 ± 0.0099    | 0.9930 ± 0.0019       |

> [!NOTE]
> While these metrics may appear elevated compared to those in the original 2020 publication, the improvements are primarily
> attributed to fine-tuned training loop conditions and hyperparameter optimization rather than fundamental architectural
> changes. The marginal gains achieved through these adjustments suggest that the original model design was already
> well-optimized for this specific segmentation task.

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

## References

- [Research Paper](http://elektron.fi.uba.ar/index.php/elektron/article/view/101/212)
- [MATLAB Implementation](https://github.com/alvgaona/heart-sounds-segmentation/tree/matlab)
