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
	Finally, summarizeResults(true, true) writes a consolidated results_summary.csv and pops up ROC and Bar charts

**C) Metrics:
1) Accuracy: overall multiclass accuracy (%)
2) AUC: aread under ROC for chosed positve class, threshold-independent
3) Thr: the score cutoff used, by default EER.
    Specific thoughts on threshold: ROC/AUC scan a full range away
    Printed FR and FA percentages do depend on threshold
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



----- Sainath Recreation -----
Window level evaluation pieplien for keyword spotting, modeled after the Sainath 14-keyword setup. Goal to take trained models, run them over sythetic streaming audio, and produce FA/hr bs FR curves for clean and noisy conditions
**A) Files:
run_sainath_window_eval.m: Entry point. Builds streams (if needed), evaluates every model, and plots the results.

make_streams_quick.m: Builds small streaming test sets:
    clean: 6 streams × 90 s (right now, 11/8)
    noisy: 6 streams × 90 s (10 dB SNR, light background)
    Saves to streams_quick/.

evaluate_window_sweep_minimal.m: Given a model and a stream set, runs sliding-window inference, aligns to the stream labels, sweeps thresholds, and writes CSVs to:
    Results/sainath/clean/curves/
    Results/sainath/noisy/curves/
It currently clips FA/h to 2000 for visibility. This is because our quick streams are short (≈9 minutes total), so even a few false alarms are normalized to a large FA/h.

plot_window_curves.m: Simple overlay plotter. Reads all sainath_curve_events_*.csv in a directory and makes an FR vs FA/h figure.

**B) How to:
1) Prepare / load config

    run_sainath_window_eval calls kws_config('sainath14') to figure out:
        where models live (cfg.paths.outputDir)
        feature geometry (25 ms, 10 ms)
        target keyword list (14 kws)

2) Create test streams (once)

    If streams_quick/streams_quick_clean.mat or streams_quick/streams_quick_noisy.mat is missing, we call make_streams_quick.
    This uses the test split to build:
        clean streams (no noise)
        noisy streams (10 dB, bgGain=0.3)
    Each stream gets 10 ms decision windows and labels.

3) Evaluate all models
For every model_*.mat in cfg.paths.outputDir:
    run clean:
        evaluate_window_sweep_minimal(modelFile, 'streams_quick_clean.mat', 'clean', [])
    run noisy:
        evaluate_window_sweep_minimal(modelFile, 'streams_quick_noisy.mat', 'noisy', [])
    this writes CSVs like:
        Results/sainath/clean/curves/sainath_curve_events_mel-only_default.csv
        Results/sainath/noisy/curves/sainath_curve_events_mel-only_default.csv

    Inside evaluate_window_sweep_minimal:
        runs slidingWindowScores on each stream WAV
        takes the max posterior over the 14 kws
        lines it up with the stream’s winLabels
        builds a threshold list from the actual scores (plus 0)
        computes FA/h using decision rate = 360000 / hour (10 ms hop)
        clips to FA/h ≤ 2000 if there are enough points
        writes the CSV

4) Plot
At the end, run_sainath_window_eval calls:
    plot_window_curves(cleanCurves, 'FR vs FA/h — CLEAN');
    plot_window_curves(noisyCurves, 'FR vs FA/h — NOISY');
Both plots show one line per model.


