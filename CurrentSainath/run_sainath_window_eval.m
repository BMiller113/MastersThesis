function run_sainath_window_eval()
% run_sainath_window_eval
% 1) ensure quick clean + noisy streams exist
% 2) evaluate all model_*.mat on clean + noisy
% 3) plot both on the SAME x-axis

    cfg = kws_config('sainath14');
    resultsDir = cfg.paths.outputDir;

    % 1) make sure streams exist
    sqDir   = fullfile(pwd, 'streams_quick');
    sqClean = fullfile(sqDir, 'streams_quick_clean.mat');
    sqNoisy = fullfile(sqDir, 'streams_quick_noisy.mat');
    if ~exist(sqClean,'file') || ~exist(sqNoisy,'file')
        make_streams_quick();
    end

    % 2) find models
    d = dir(fullfile(resultsDir, 'model_*.mat'));
    if isempty(d)
        error('No model_*.mat in %s', resultsDir);
    end

    % prepare curve dirs
    cleanCurves = fullfile(resultsDir, 'sainath', 'clean', 'curves');
    noisyCurves = fullfile(resultsDir, 'sainath', 'noisy', 'curves');
    if ~exist(cleanCurves,'dir'), mkdir(cleanCurves); end
    if ~exist(noisyCurves,'dir'), mkdir(noisyCurves); end
    delete(fullfile(cleanCurves, 'sainath_curve_events_*.csv'));
    delete(fullfile(noisyCurves, 'sainath_curve_events_*.csv'));

    % 3) run eval
    for k = 1:numel(d)
        mf = fullfile(d(k).folder, d(k).name);
        fprintf('\n=== Evaluating %s (CLEAN) ===\n', d(k).name);
        evaluate_window_sweep_minimal(mf, sqClean, 'clean', []);

        fprintf('=== Evaluating %s (NOISY) ===\n', d(k).name);
        evaluate_window_sweep_minimal(mf, sqNoisy, 'noisy', []);
    end

    % 4) plot both, same x-axis
    fixedX = 2000;  % <-- change to 2,5 here for full sainath
    plot_window_curves(cleanCurves, 'FR vs FA/h — CLEAN', fixedX);
    plot_window_curves(noisyCurves, 'FR vs FA/h — NOISY', fixedX);
end
