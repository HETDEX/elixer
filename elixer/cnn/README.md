#from Shiro's 
#/work/10359/shiromukae/mysharedirectory/share/cnn_20260316/

# CNN Model Fitting Code

This repository contains the code used to apply trained CNN models to 2D spectral images and generate CNN scores for each `detectid`.

## Overview

The code applies a trained CNN model to 2D spectra stored as NumPy `.npy` files. It processes spectra from HDR5, HDR4, and HDR3 directories, applies trained CNN models from 3-fold cross-validation, and outputs CNN score catalogs.

The CNN architecture used here is:

```text
TDSA_LeakyGAP_logit
```

where:

- `TDSA` stands for **Two-Dimensional Spectrum Architecture**.
- `LeakyGAP` indicates that the model uses **LeakyReLU** and **Global Average Pooling**.
- `logit` means that the model returns logits, not probabilities.

The logits are converted to CNN scores using the sigmoid function during model fitting.

## 1. Input files and directories

The input 2D spectra should be stored as NumPy `.npy` files.

Each file should be named using its `detectid`:

```text
{detectid}.npy
```

For example:

```text
1234567890.npy
```

Only the weighted-sum 2D spectrum is used as the CNN input. In the original data structure, this corresponds to:

```python
['im_sum']
```

The `['im_sum']` array was directly converted to a `.npy` file for each `detectid`.

The base input directory can be set to any directory containing the HDR subdirectories. For example:

```text
<base_input_dir>/
```

The code expects the following HDR subdirectories under the base input directory:

```text
hdr5/
hdr4/
hdr3/
```

The expected input directory structure is:

```text
<base_input_dir>/
├── hdr5/
│   ├── {detectid}.npy
│   └── ...
├── hdr4/
│   ├── {detectid}.npy
│   └── ...
└── hdr3/
    ├── {detectid}.npy
    └── ...
```

The base input directory is specified in `model_fitting_hdr543.py`:

```python
input_dir = '<base_input_dir>/'
```

Each `.npy` file is loaded as a 2D weighted-sum spectrum. During data loading, the spectrum is normalized by its mean and standard deviation. The spatial cutout is applied during model fitting, not when preparing the `.npy` files.

## 2. Main files

```text
model_fitting_hdr543.py
```

Main script for applying the trained CNN model to HDR5, HDR4, and HDR3 spectra. The final score is calculated as the mean CNN score from the three cross-validation folds.

```text
model_fitting_config.py
```

Configuration and utility script. It defines the model ID, model path, input size, output directory, dataset class, and model fitting function.

```text
TDSA_LeakyGAP_logit.py
```

CNN model architecture. The model returns logits, which are converted to probabilities during model fitting.

## 3. Model files

The trained model parameters are loaded from:

```text
./model/
```

The training ID is defined in `model_fitting_config.py`:

```python
training_id = '14.0.1_d0.2'
```

Because the model was trained using 3-fold cross-validation, the following three model files are expected:

```text
./model/best_model_{training_id}_fold1.pth
./model/best_model_{training_id}_fold2.pth
./model/best_model_{training_id}_fold3.pth
```

The model is loaded using the same architecture as used during training. 


## 4. How to run the code

To run the model fitting code, execute:

```bash
python model_fitting_hdr543.py
```

The script processes the HDR directories in the following order:

```text
hdr5 → hdr4 → hdr3
```

For each HDR directory, the code:

1. Collects all `.npy` files in the directory.
2. Loads each trained model from the three cross-validation folds.
3. Applies the CNN model to all spectra.
4. Saves the CNN scores for each `detectid`.
5. Averages the CNN scores over the three folds.

The input data are processed in batches to reduce memory usage.

## 5. Notes on running on Stampede3

The code automatically uses a GPU if CUDA is available. Otherwise, it runs on CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

I initially tried to use a GPU on Stampede3, but for an unknown reason it did not work properly. Therefore, I submitted the jobs using CPUs.

The CPU runtime was still reasonable: processing approximately 1.6 million spectra took less than one hour.

## 6. Output files and directories

The output directory is automatically created as:

```text
./output/{training_id}/
```

For each HDR catalog, the code outputs a table:

```text
cnn_{training_id}_hdr5.txt
cnn_{training_id}_hdr4.txt
cnn_{training_id}_hdr3.txt
```

Each HDR-level output table contains:

```text
detectid
CNN_Score_2D_Spectra_Fold_1
CNN_Score_2D_Spectra_Fold_2
CNN_Score_2D_Spectra_Fold_3
CNN_Score_2D_Spectra
```

where:

- `detectid` is extracted from the input file name.
- `CNN_Score_2D_Spectra_Fold_1` is the CNN score from fold 1.
- `CNN_Score_2D_Spectra_Fold_2` is the CNN score from fold 2.
- `CNN_Score_2D_Spectra_Fold_3` is the CNN score from fold 3.
- `CNN_Score_2D_Spectra` is the average CNN score over the three folds.

Finally, the HDR5, HDR4, and HDR3 results are stacked into one combined catalog:

```text
cnn_{training_id}_hdr5.0.1_lae.txt
```

This final output file contains only:

```text
detectid
CNN_Score_2D_Spectra
```

The final table is sorted by `detectid`.
