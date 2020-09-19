# Heart Sounds Segmentation

Bachelor thesis for Electronics Engineering at Faculty of Engineering, University of Buenos Aires.

## Requirements

- MATLAB R2019b or newer

## Datasets

The dataset used in this project is the one provided by [David Springer].
Additionally, it has automatically extracted labels using [Springer's labeling algorithm][Labeling algorithm].

The dataset can be downloaded from [here][Dataset] in MAT format.

It contains the following fields.

- `audio_data`: raw audio data
- `labels`: label for each sample.

```
1: Sound 1
2: Systolic interval
3: Sound 2
4: Diastolic interval
```

- `annotations`: R-wave and end of T-wave annotations.
- `binary_diagnosis`: 0 if it's normal or 1 if it's abnormal.
- `patient_number`: patient number associated with audio data by position in the array.
- `features`: Hilbert envelope, Homomorphic envelope, DWT envelope, PSD envelope.

## Results

_K-Fold Cross-validation_ has been used to obtain results. 
Where `K = 10`.

After that, only one model is selected by taking into consideration the _Area Under the Curve_ (AUC) and
metrics such as _Precision_, _Sensitivity_, _F1-score_ and _Accuracy_.

Fold number 1 turned out to be the best one.
Below you can see the _Receiver Operation Characteristic_ (ROC) curve and its confusion matrix.

### Training progress

The training was performed on a validation set to control the losses.
Preventing the model from overfitting.

![Training Progress]

```
|======================================================================================================================|
|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  |  Base Learning  |
|         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       |
|======================================================================================================================|
|       1 |           1 |       00:03:33 |       23.81% |       54.82% |       1.3868 |       1.6246 |          0.0100 |
|       1 |          50 |       00:36:26 |       68.17% |       79.93% |       0.7300 |       0.6020 |          0.0100 |
|       1 |         100 |       01:08:00 |       79.32% |       87.75% |       0.5484 |       0.3595 |          0.0100 |
|       1 |         150 |       01:38:52 |       85.29% |       89.86% |       0.4015 |       0.2817 |          0.0100 |
|       1 |         200 |       02:20:33 |       85.14% |       84.66% |       0.4000 |       0.4206 |          0.0100 |
|       2 |         250 |       02:46:28 |       88.87% |       91.66% |       0.3009 |       0.2323 |          0.0100 |
|       2 |         300 |       03:26:02 |       90.89% |       91.93% |       0.2512 |       0.2306 |          0.0100 |
|       2 |         350 |       03:52:03 |       89.28% |       91.61% |       0.2943 |       0.2364 |          0.0100 |
|       2 |         400 |       04:17:42 |       89.53% |       92.49% |       0.2711 |       0.2140 |          0.0100 |
|       3 |         450 |       04:48:48 |       86.87% |       92.67% |       0.3368 |       0.2074 |          0.0100 |
|       3 |         500 |       05:30:56 |       90.88% |       93.18% |       0.2398 |       0.1887 |          0.0100 |
|       3 |         550 |       05:59:04 |       89.65% |       92.69% |       0.2736 |       0.2114 |          0.0100 |
|       3 |         600 |       06:41:37 |       92.79% |       92.84% |       0.1917 |       0.2019 |          0.0100 |
|       3 |         650 |       07:33:08 |       91.32% |       92.85% |       0.2366 |       0.2066 |          0.0100 |
|       4 |         700 |       08:19:27 |       92.58% |       93.73% |       0.1957 |       0.1856 |          0.0010 |
|       4 |         750 |       08:44:36 |       92.41% |       93.88% |       0.2048 |       0.1797 |          0.0010 |
|       4 |         800 |       09:21:05 |       93.42% |       93.86% |       0.1695 |       0.1804 |          0.0010 |
|       4 |         850 |       10:12:13 |       94.16% |       93.57% |       0.1525 |       0.1876 |          0.0010 |
|       5 |         900 |       10:54:25 |       93.85% |       93.88% |       0.1601 |       0.1791 |          0.0010 |
|       5 |         950 |       11:24:33 |       94.71% |       93.77% |       0.1439 |       0.1830 |          0.0010 |
|       5 |        1000 |       12:13:00 |       95.13% |       93.79% |       0.1266 |       0.1877 |          0.0010 |
|       5 |        1010 |       12:21:20 |       94.45% |              |       0.1483 |              |          0.0010 |
|======================================================================================================================|
```

### Confusion Matrix

The actual confision matrix is the 4x4 matrix. 
In addition to it on the right side you can find the sensitivity or recall for each class.
Below, the precision is computed for each class as well.

![Confusion Matrix]

Off of the confusion matrix the accuracy can also be computed.

- ACC = 93.29%   

### ROC Curve

In this plot you can find the ROC curve for each class and the AUC scores are computed.

![ROC Curve]

- AUC1: 0.984
- AUC2: 0.992
- AUC3: 0.994
- AUC4: 0.992
- Average: 0.991

### Model

The model has been saved for future use and can be found in [here][Network].

### Contributors

- Advisor: Dr. Eng. Pedro David Arini
- Student: Álvaro Joaquín Gaona

[David Springer]: https://github.com/davidspringer
[Labeling algorithm]: https://github.com/davidspringer/Springer-Segmentation-Code/blob/master/labelPCGStates.m
[Dataset]: https://ag-datasets-89f203ac-44ed-4a06-9395-1e069e8e662d.s3-us-west-2.amazonaws.com/springer_dataset.mat

[Confusion Matrix]: images/2020-06-29-T01-22-02/testing-cm-1.png 
[Training Progress]: images/2020-06-29-T01-22-02/training-progress-1.png
[ROC Curve]: images/2020-06-29-T01-22-02/roc-1.png
[Network]: resources/models/net-1.mat
