# Heart Sounds Segmentation

Bachelor thesis from University of Buenos Aires.

**Thesis tutors**: Dr. Eng. Pedro David Arini and Dr. Eng. Maria Paula Bonomini. 

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


[David Springer]: https://github.com/davidspringer
[Labeling algorithm]: https://github.com/davidspringer/Springer-Segmentation-Code/blob/master/labelPCGStates.m
[Dataset]: https://ag-datasets-89f203ac-44ed-4a06-9395-1e069e8e662d.s3-us-west-2.amazonaws.com/springer_dataset.mat
