function summarizeResults(exportCSV, makePlots)

    if nargin < 1, exportCSV = false; end
    if nargin < 2, makePlots = false; end

    cfg = [];
    if exist('kws_config','file')
        try
            cfg = kws_config();
        catch

        end
    end

    % Search for directory to check
    resultsDir = '.';
    if ~isempty(cfg) && isfield(cfg,'paths') && isfield(cfg.paths,'outputDir') && ~isempty(cfg.paths.outputDir)
        resultsDir = cfg.paths.outputDir;
    end

    % Plotting Preferences
    xAxisMode   = 'fpr_percent';   % 'fpr_percent' | 'fpr' | 'fa_per_hour' (utterance ROC -> use fpr_percent)
    addDET      = false;           % add DET?
    xlimROC     = [];              % [0 5] or similar for percent mode
    ylimROC     = [];              % [0 20] or similar percent mode
    hopSec      = 0.01;            % only relevant for 'fa_per_hour' (streaming use-case)

    if ~isempty(cfg) && isfield(cfg,'plots') && isfield(cfg.plots,'roc')
        pr = cfg.plots.roc;
        if isfield(pr,'xAxis') && ~isempty(pr.xAxis), xAxisMode = pr.xAxis; end
        if isfield(pr,'addDET') && ~isempty(pr.addDET), addDET = logical(pr.addDET); end
        if isfield(pr,'xlim') && ~isempty(pr.xlim), xlimROC = pr.xlim; end
        if isfield(pr,'ylim') && ~isempty(pr.ylim), ylimROC = pr.ylim; end
        if isfield(cfg.plots,'frameHopSec') && ~isempty(cfg.plots.frameHopSec)
            hopSec = cfg.plots.frameHopSec;
        end
    end

    % Scan directory
    files = dir(fullfile(resultsDir,'results_*.mat'));

    Mode = {};
    GenderMode = {};
    MelMode = {};
    FilterGender = {};
    Accuracy = [];
    FR = [];
    FA = [];
    rocCurves = {};
    legendLabels = {};

    for k = 1:numel(files)
        fpath = fullfile(files(k).folder, files(k).name);
        S = load(fpath);
        if ~isfield(S,'results') || ~isfield(S,'rocInfo')
            warning('Skipping %s: missing results or rocInfo', files(k).name);
            continue;
        end
        r = S.results;
        if ~isfield(r,'GenderMode') || ~isfield(r,'MelMode')
            warning('Invalid format in %s. Skipping.', files(k).name);
            continue;
        end

        % Determine gender group
        fg = 'all';
        if isfield(r,'FilterGender')
            fg = r.FilterGender;
        else
            % Try infer from filename
            tok = regexp(files(k).name, '(male|female)','match','once');
            if ~isempty(tok), fg = tok; end
        end

        label = sprintf('%s - %s - %s', upper(r.GenderMode), r.MelMode, fg);
        Mode{end+1,1}         = label;
        GenderMode{end+1,1}   = r.GenderMode;
        MelMode{end+1,1}      = r.MelMode;
        FilterGender{end+1,1} = fg;
        Accuracy(end+1,1)     = double(r.Accuracy);
        FR(end+1,1)           = double(r.FR);
        FA(end+1,1)           = double(r.FA);

        if isfield(S,'rocInfo') && isfield(S.rocInfo, 'far') && ~isempty(S.rocInfo.far)
            rocCurves{end+1}    = S.rocInfo;
            legendLabels{end+1} = label;
        end
    end

    % Build summary table
    FRpercent = FR;
    FApercent = FA;
    T = table(Mode, GenderMode, MelMode, FilterGender, Accuracy, FRpercent, FApercent);
    disp(T);

    % Write CSVs to the same directory as the results
    if exportCSV
        writetable(T, fullfile(resultsDir,'results_summary_all.csv'));
        if any(strcmpi(T.FilterGender,'male'))
            writetable(T(strcmpi(T.FilterGender,'male'),:),   fullfile(resultsDir,'results_summary_male.csv'));
        end
        if any(strcmpi(T.FilterGender,'female'))
            writetable(T(strcmpi(T.FilterGender,'female'),:), fullfile(resultsDir,'results_summary_female.csv'));
        end
        disp('Exported results_summary_all.csv (+ male/female CSVs if present)');
    end

    % Bar chart (optional)
    if makePlots && ~isempty(T)
        figure('Visible','on');
        bar(categorical(T.Mode), [T.Accuracy, T.FRpercent, T.FApercent]);
        ylabel('Percentage');
        legend({'Accuracy','False Reject','False Alarm'});
        title('Performance Comparison by Mode');
        xtickangle(30);
        grid on;
    end

    % ROC curves (optional)  -- 8/29 -> use FPR/FRR percent on linear axes by default
    if makePlots && ~isempty(rocCurves)
        figure('Visible','on'); hold on;

        for i = 1:numel(rocCurves)
            rc = rocCurves{i};
            if isempty(rc.far) || isempty(rc.frr), continue; end

            switch lower(xAxisMode)
                case 'fpr_percent'
                    x = rc.far * 100;    % False Positive Rate (%)
                    y = rc.frr * 100;    % False Reject Rate (%)
                    plot(x, y, 'LineWidth', 2);

                case 'fpr'  % raw 0..1 rates
                    x = rc.far;
                    y = rc.frr;
                    plot(x, y, 'LineWidth', 2);

                case 'fa_per_hour'
                    % NOTE: This only makes sense for streaming ROC computed over time.
                    % Makes no sense for utterance based ROC
                    if hopSec <= 0
                        warning('summarizeResults:InvalidHop', ...
                                'frameHopSec <= 0; falling back to FPR%% axis.');
                        x = rc.far * 100; y = rc.frr * 100;
                        plot(x, y, 'LineWidth', 2);
                    else
                        % If FAR is per-decision, FA/hour ~= FPR * (decisions/hour).
                        % decisions/hour = 3600 / hopSec (if decisions every hop).
                        x = rc.far * (3600 / hopSec);
                        y = rc.frr * 100;
                        semilogx(x, y, 'LineWidth', 2);  % keep log scale for "per hour"
                    end

                otherwise
                    % Fallback to percent axis
                    x = rc.far * 100; y = rc.frr * 100;
                    plot(x, y, 'LineWidth', 2);
            end
        end

        % Labels and default zooms
        switch lower(xAxisMode)
            case 'fpr_percent'
                xlabel('False Positive Rate (%)');
                ylabel('False Reject Rate (%)');
                if ~isempty(xlimROC), xlim(xlimROC); end
                if ~isempty(ylimROC), ylim(ylimROC); end
            case 'fpr'
                xlabel('False Positive Rate');
                ylabel('False Reject Rate');
                if ~isempty(xlimROC), xlim(xlimROC); end
                if ~isempty(ylimROC), ylim(ylimROC); end
            case 'fa_per_hour'
                xlabel('False Alarms per Hour');
                ylabel('False Reject Rate (%)');
                % User-provided limits, otherwise leave auto
                if ~isempty(xlimROC), xlim(xlimROC); end
                if ~isempty(ylimROC), ylim(ylimROC); end
        end

        title('ROC Curves by Mode');
        if ~isempty(legendLabels)
            legend(legendLabels, 'Location','best');
        end
        grid on;
    end

    % Optional DET plot (spreads out the bottom-left corner)
    if makePlots && ~isempty(rocCurves) && addDET
        figure('Visible','on'); hold on;
        for i = 1:numel(rocCurves)
            rc = rocCurves{i};
            if isempty(rc.far) || isempty(rc.frr), continue; end
            xd = norminv_clip(rc.far);
            yd = norminv_clip(rc.frr);
            plot(xd, yd, 'LineWidth', 2);
        end
        xlabel('False Positive (normal deviate)');
        ylabel('False Negative (normal deviate)');
        title('DET Curves by Mode');
        if ~isempty(legendLabels)
            legend(legendLabels, 'Location','best');
        end
        grid on;
    end
end

% Helper: safe inverse normal CDF (avoids Inf at 0/1)
function z = norminv_clip(p)
    p = min(max(p, 1e-6), 1-1e-6);
    z = -sqrt(2) * erfcinv(2*p);  % same as norminv(p)
end
