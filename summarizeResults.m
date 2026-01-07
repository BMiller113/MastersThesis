function summarizeResults(exportCSV, makePlots, searchDir)
% summarizeResults
% Scans results_*.mat, builds a summary table, optionally writes CSVs,
%   summarizeResults()                      % no CSV, no plots, current folder
%   summarizeResults(true, true)            % CSV + plots, current folder
%   summarizeResults(true, false, 'Results')% CSV only, specific folder

    if nargin < 1 || isempty(exportCSV), exportCSV = false; end
    if nargin < 2 || isempty(makePlots), makePlots = false; end
    if nargin < 3 || isempty(searchDir), searchDir = pwd; end

    files = dir(fullfile(searchDir, 'results_*.mat'));
    if isempty(files)
        warning('summarizeResults:NoFiles', 'No results_*.mat found in %s.', searchDir);
    end

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
        S = load(fullfile(files(k).folder, files(k).name));
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
        if isfield(r,'FilterGender') && ~isempty(r.FilterGender)
            fg = r.FilterGender;
        else
            % Try infer from filename (supports older runs)
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

    % Rename headers
    FRpercent = FR;
    FApercent = FA;
    T = table(Mode, GenderMode, MelMode, FilterGender, Accuracy, FRpercent, FApercent);
    disp(T);

    % ---- CSV export ----
    if exportCSV
        try
            writetable(T, fullfile(searchDir, 'results_summary_all.csv'));
            if any(strcmpi(T.FilterGender,'male'))
                writetable(T(strcmpi(T.FilterGender,'male'),:), ...
                    fullfile(searchDir, 'results_summary_male.csv'));
            end
            if any(strcmpi(T.FilterGender,'female'))
                writetable(T(strcmpi(T.FilterGender,'female'),:), ...
                    fullfile(searchDir, 'results_summary_female.csv'));
            end
            fprintf('Exported results_summary_*.csv to %s\n', searchDir);
        catch ME
            warning('summarizeResults:CSV', 'Failed to write CSVs: %s', ME.message);
        end
    end

    % Plots
    if makePlots && height(T) > 0
        % Bar chart
        try
            figure('Visible','on');
            bar(categorical(T.Mode), [T.Accuracy, T.FRpercent, T.FApercent]);
            ylabel('Percentage');
            legend({'Accuracy','False Reject','False Alarm'}, 'Location','best');
            title('Performance Comparison by Mode');
            xtickangle(30);
            grid on;
        catch ME
            warning('summarizeResults:BarPlot', 'Bar plot failed: %s', ME.message);
        end
    end

    if makePlots && ~isempty(rocCurves)
        % ROC: FAR (per frame) -> FA/hr or FPR%, depending on how rocInfo.far was computed.
        try
            figure('Visible','on'); hold on;
            plotted = false;
            for i = 1:numel(rocCurves)
                rc = rocCurves{i};
                if isempty(rc.far) || isempty(rc.frr), continue; end
                % Default rendering: FPR% vs FRR% 
                x = rc.far * 100;     
                y = rc.frr * 100;  
                plot(x, y, 'LineWidth', 2);
                plotted = true;
            end
            if plotted
                xlabel('False Positive Rate (%)'); ylabel('False Reject Rate (%)');
                title('ROC Curves by Mode');
                legend(legendLabels, 'Location','best');
                grid on;
            else
                title('ROC Curves (none available)'); axis off;
            end
        catch ME
            warning('summarizeResults:ROCPlot', 'ROC plot failed: %s', ME.message);
        end
    end
end
