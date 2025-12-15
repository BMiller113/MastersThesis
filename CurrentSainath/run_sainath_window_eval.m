function run_sainath_window_eval()
% Run Sainath-style window FR vs FA/hour evaluation for all models
% and plot clean/noisy curves.
%
% IMPORTANT: This version assumes you have already done:
%   cd('C:\Users\bjren\MATLAB\Projects\KeywordSpottingThesis\GenderAnalysis')
% and that:
%   streams_long\streams_long_clean.mat
%   streams_long\streams_long_noisy.mat
% live under that folder.

    fprintf('run_sainath_window_eval: using LONG streams.\n');

    % Use the CURRENT WORKING DIRECTORY (GenderAnalysis)
    rootDir = pwd;

    cfg = kws_config('sainath14'); %#ok<NASGU>

    % ----- streams -----
    streamsDir = fullfile(rootDir, 'streams_long');
    cleanMat   = fullfile(streamsDir, 'streams_long_clean.mat');
    noisyMat   = fullfile(streamsDir, 'streams_long_noisy.mat');

    if ~isfile(cleanMat) || ~isfile(noisyMat)
        error(['streams_long .mat files are missing. Expected:' newline ...
               '  %s' newline ...
               '  %s'], cleanMat, noisyMat);
    end

    fprintf('  CLEAN MAT: %s\n', cleanMat);
    fprintf('  NOISY MAT: %s\n', noisyMat);

    % ----- models -----
    % Assumes your model .mat files are under GenderAnalysis\Results
    modelsDir = fullfile(rootDir, 'Results');
    modelFiles = { ...
        fullfile(modelsDir, 'model_mel-only_default.mat'), ...
        fullfile(modelsDir, 'model_mel-only_narrow.mat'), ...
        fullfile(modelsDir, 'model_mel-only_prop7k.mat'), ...
        fullfile(modelsDir, 'model_mel-only_wide.mat'), ...
        fullfile(modelsDir, 'model_none_default.mat') ...
    };

    % sanity-check models
    for i = 1:numel(modelFiles)
        if ~isfile(modelFiles{i})
            error('Missing model file: %s', modelFiles{i});
        end
    end

    % ----- evaluate all models on clean + noisy -----
    for i = 1:numel(modelFiles)
        mf = modelFiles{i};
        mname = get_model_name(mf);

        fprintf('\n=== Evaluating %s (CLEAN) ===\n', mname);
        % Your evaluate_window_sweep_minimal should be writing
        % Results\sainath\clean\curves\sainath_curve_events_transFA100ms_*.csv
        evaluate_window_sweep_minimal(mf, cleanMat, 'clean', []);

        fprintf('=== Evaluating %s (NOISY) ===\n', mname);
        % Likewise for noisy
        evaluate_window_sweep_minimal(mf, noisyMat, 'noisy', []);
    end

    % ----- plot all curves (clean + noisy) -----
    try
        plot_window_curves();
    catch ME
        warning('plot_window_curves failed: %s', ME.message);
    end

    fprintf('run_sainath_window_eval: done.\n');
end

function name = get_model_name(path)
    [~,bn,~] = fileparts(path);
    name = bn;
end
