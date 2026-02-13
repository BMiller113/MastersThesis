Keyword Spotting Thesis - Last Updated 2/12

This repository contains MATLAB code for running reproducible keyword
spotting experiments on the Google Speech Commands dataset
(Versions 1 and 2).

The goal of this project is to compare lightweight CNN architectures
under various preprocessing and evaluation conditions,
and to report detection performance using ROC/AUC metrics alongside
classification accuracy and computational cost.


1) Project Overview

For a given dataset (V1 or V2) and configuration (architecture + feature
shape), the system:

1.  Loads audio files from the Speech Commands dataset.
2.  Extracts log-mel spectrogram features into a fixed-size tensor:
    [freqBins × timeFrames × 1 × N]
3.  Trains a lightweight CNN classifier.
4.  Evaluates the trained model on a held-out test split.
5.  Exports:
    -   Model file (.mat)
    -   Metrics (.mat and .csv)
    -   ROC plots (.fig and .png)
    -   Overlay ROC comparisons


2) Required Toolboxes

-   Deep Learning Toolbox (required)
-   Audio Toolbox (strongly recommended)

If melSpectrogram is unavailable, fallback logic may be used, but Audio
Toolbox is strongly recommended for consistent reproduction.


3) Dataset Requirements

Expected folder structure:

Kaggle_GoogleSpeechCommandsV2/ yes/.wav no/.wav … background_noise/*.wav
testing_list.txt validation_list.txt

The same structure applies to V1.

The dataset folder name must match exactly what is specified
in kws_config.m.


4) Configuration (kws_config)

All experiment settings are centralized in kws_config.m.

Key settings:

Dataset Paths: cfg.paths.datasetRootV2, cfg.paths.datasetRootV1,
cfg.paths.outputDir

Feature Geometry: cfg.features.baseBands (e.g., 40, 80),
cfg.features.targetFrames (e.g., 32 or 98), cfg.features.frameMs,
cfg.features.hopMs

Architecture: cfg.model.arch ‘trad-fpool3’, ‘tpool2’, ‘one-fstride4’

Plotting Controls: cfg.runtime.makePlots, cfg.runtime.figureVisibility


5) Running the Project

Step 1: Verify dataset paths/desired configs in kws_config.m

Step 2: Open MATLAB in project root

Step 3: Add project to path addpath(genpath(pwd)); savepath;

Step 4: Run main.m

Results will be written to: 
Results/ models/ 
Results/metrics/ 
Results/ROC/overlay/


6) Key Terminology
Keyword Spotting (KWS): Small-footprint neural networks that detect
short spoken keywords.

Log-Mel Spectrogram: 2D representation of audio with frequency bins
(vertical) and time frames (horizontal).

freqBins: Number of mel frequency bands.

timeFrames: Number of time steps fed into CNN.

MACs: Multiply-accumulate operations for CNN forward pass only.

Params: Total learnable parameters (weights + biases).

ROC Curve: True Positive Rate vs False Positive Rate across thresholds.

AUC: Area Under ROC Curve (1.0 = perfect separation).


7) Output Artifacts

models/model_.mat

metrics/metrics_.mat metrics_.csv

ROC/ROC_.fig ROC_.png

ROC/overlay/ROC_OVERLAY__.fig


8) Interpreting Results

Top-1 Accuracy: Classification sanity check metric, indicated the model is
learning meaningful patterns.

ROC/AUC: Primary detection metric for keyword spotting, low false alram rate
indicating good performance.

MACs: Computational cost of CNN forward pass, useful for cost estimation 
(does not include preprocessing in calculation).

Params: Model size indicator (weights + biases), total nummber of learnable
parameters.


() Troubleshooting

Dataset folder not found: Check datasetRoot paths.

Incorrect input size error: Model was trained on different feature
geometry.

ROC not generating: Verify plotting flags in config.


10) Quick Workflow

1.  Run selected architectures.
2.  Inspect metrics CSV.
3.  Compare ROC overlays.
4.  Choose based on ROC, MACs, and accuracy sanity.
