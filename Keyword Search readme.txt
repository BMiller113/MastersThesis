Keyword Search readme --Last Updated 8/25/2025


**A) Requirements: 
MATLAB R2024b
Toolboxes: Deep Learning, Audio, Signal Processing, Statistics and Machine Learning Toolbox

Dataset:
Google Speech Commands V2
Dataset folder must contain:
testing_list.txt, validation_list.txt, class subfolders with .wav files

	Example:
	<DATASET_ROOT>/
  		testing_list.txt
  		validation_list.txt
  		bed/xyz.wav
  		yes/xyz.wav
  		...

**B) Quick Start:
1) Open kws_config and set paths, UI runtime toggles, choose experiments, and model/training
2) Build gender map (one time). This runs a pitch based heuristic on at least one file per speaker to label male/female

	datasetRoot = kws_config().paths.datasetRoot;
	genderMap = build_speaker_gender_map(datasetRoot);
	save('speakerGenderMap.mat','genderMap');

3) Run main
	Data is loaded and filtered per experiment
	Features are extracted (mel or linear, depending on mode)
	CNN is training with a stratified holdout split
	Model is evaluated
	Results are saved to cfg.paths.outputDIR
	Finally, summarizeResults(true, true) writes a consolidated results_summary.csv and pops up ROC and Bar 	charts

**C) Metrics:
1) Accuracy: overall multiclass accuracy (%)
2) AUC: aread under ROC for chosed positve class, threshold-independent
3) Thr: the score cutoff used, by default EER.
4) FR (%): among true positives, percent that system misses Thr
5) FA (%): among negatives, percent system fires on at Thr
6) Support: count of examples used in a given metric 

**D) Troubleshooting:
Dataset not found: check cfg.paths.datasetRoot
Gender map missing: run the onetime builder, if error persists, ensure main is being run where the 				    speakerGenderMap.mat is visible to it
Too many warnings during extraction: set cfg.runtime.suppressWarnings = true
Class mismatch errors: ensure you're using latest evaluateModel.m. Matlab can cache an older version, to clear run:

	clear evaluateModel; rehash toolboxcache
	which -all evaluateModel                    

**E) Notes on frontend modes:
default / narrow / wide – fixed 40 Mel bands (by default).

prop7k / prop8k – extend the upper frequency to ~7–8 kHz and increase the number of Mel bands proportionally.

linear – linearly spaced triangular filter bank (female-only toggle). Good to test claims about linear vs Mel for female voices.

