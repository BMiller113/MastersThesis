function summarizeResults(exportCSV, makePlots)
    if nargin < 1, exportCSV = false; end
    if nargin < 2, makePlots = false; end

    files = dir('results_*.mat');

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
        S = load(files(k).name);
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
            % Try infer from filename (supports older runs saved as ..._male.mat / ..._female.mat)
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
            rocCurves{end+1}   = S.rocInfo;
            legendLabels{end+1} = label;
        end
    end


    FRpercent = FR;
    FApercent = FA;
    T = table(Mode, GenderMode, MelMode, FilterGender, Accuracy, FRpercent, FApercent);
    disp(T);

    if exportCSV
        writetable(T, 'results_summary_all.csv');
        if any(strcmpi(T.FilterGender,'male'))
            writetable(T(strcmpi(T.FilterGender,'male'),:), 'results_summary_male.csv');
        end
        if any(strcmpi(T.FilterGender,'female'))
            writetable(T(strcmpi(T.FilterGender,'female'),:), 'results_summary_female.csv');
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

    % ROC curves (optional)
    if makePlots && ~isempty(rocCurves)
        figure('Visible','on'); hold on;
        for i = 1:numel(rocCurves)
            rc = rocCurves{i};
            if isempty(rc.far) || isempty(rc.frr), continue; end
            semilogx(rc.far * 3600 / 0.01, rc.frr * 100, 'LineWidth', 2);
        end
        xlabel('False Alarms per Hour'); ylabel('False Reject Rate (%)');
        title('ROC Curves by Mode'); legend(legendLabels, 'Location','best'); 
        grid on;
    end
end
