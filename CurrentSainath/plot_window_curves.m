function plot_window_curves()
% plot_window_curves
%   Read Sainath-style curve CSVs from:
%       Results\sainath\clean\curves
%       Results\sainath\noisy\curves
%   Using ONLY the *events* files:
%       sainath_curve_events_transFA100ms_*.csv
%   and plot ONE (smoothed) line per model configuration.
%
%   This is intentionally "simple and clean":
%     - No eventCenter overlays
%     - No multiple variants per model
%     - No aggressive filtering that can hide curves
%     - One legend entry per model
%
%   It keeps the general shape of your 11-20-2025 curves but makes them
%   less messy: single line per mode, sorted by FA/hr, lightly smoothed.

    rootDir = pwd;
    baseDir = fullfile(rootDir, 'Results', 'sainath');

    conditions = {'clean','noisy'};

    % These should match your model suffixes in the CSV filenames
    modelSuffixes = { ...
        'mel-only_default', ...
        'mel-only_narrow', ...
        'mel-only_prop7k', ...
        'mel-only_wide', ...
        'none_default' ...
    };

    % Pretty names for the legend (optional)
    modelNames = { ...
        'mel-only default', ...
        'mel-only narrow', ...
        'mel-only 7k', ...
        'mel-only wide', ...
        'none default' ...
    };

    % Smoothing + decimation parameters
    maxPoints   = 800;   % limit number of points per curve (visual clarity)
    smoothWidth = 25;    % moving-average window (in index samples)

    for c = 1:numel(conditions)
        cond = conditions{c};
        curvesDir = fullfile(baseDir, cond, 'curves');

        if ~isfolder(curvesDir)
            warning('plot_window_curves: curves folder not found for condition %s: %s', ...
                    cond, curvesDir);
            continue;
        end

        figure('Name', sprintf('Sainath-style curves (%s)', cond), ...
               'NumberTitle', 'off');
        hold on;
        somethingPlotted = false;

        for k = 1:numel(modelSuffixes)
            suffix = modelSuffixes{k};
            prettyName = modelNames{k};

            % We use ONLY the "events" curves here
            csvFile = fullfile(curvesDir, ...
                sprintf('sainath_curve_events_transFA100ms_%s.csv', suffix));

            if ~isfile(csvFile)
                warning('plot_window_curves: missing file for %s (%s): %s', ...
                        cond, suffix, csvFile);
                continue;
            end

            T = readtable(csvFile);

            if ~all(ismember({'FA_per_hr','FR_fraction'}, T.Properties.VariableNames))
                warning('plot_window_curves: file %s missing expected columns FA_per_hr, FR_fraction', csvFile);
                continue;
            end

            fa = T.FA_per_hr;
            fr = T.FR_fraction;

            % Remove NaNs and Infs
            bad = isnan(fa) | isnan(fr) | isinf(fa) | isinf(fr);
            fa(bad) = [];
            fr(bad) = [];

            if numel(fa) < 2
                warning('plot_window_curves: too few points for %s (%s)', cond, suffix);
                continue;
            end

            % Sort by FA_per_hr (left-to-right curve)
            [fa, idx] = sort(fa(:));
            fr = fr(idx);

            % Optional decimation: keep at most maxPoints, spread evenly
            n = numel(fa);
            if n > maxPoints
                keepIdx = round(linspace(1, n, maxPoints));
                fa = fa(keepIdx);
                fr = fr(keepIdx);
            end

            % Light smoothing in FR (moving average)
            if smoothWidth > 1 && numel(fr) > smoothWidth
                fr = movmean(fr, smoothWidth);
            end

            % Plot the curve (single line, no markers)
            plot(fa, fr, 'LineWidth', 1.5, 'DisplayName', prettyName);
            somethingPlotted = true;
        end

        if ~somethingPlotted
            warning('No valid curves were plotted for condition: %s', cond);
            close(gcf);
            continue;
        end

        % Axes + formatting
        grid on;
        xlabel('False alarms per hour');
        ylabel('Fraction of events missed (FR)');
        title(sprintf('Keyword Spotting FR vs FA/hr (%s)', cond));

        % Let the x-axis go up to whatever your data supports, but cap a bit
        xlim auto;
        xl = xlim;
        if xl(2) > 10
            xlim([0 10]);   % don’t go crazy wide on FA axis
        elseif xl(2) <= 0
            xlim([0 1]);
        end

        % Show full FR 0–1 (don’t hide “bad” performance)
        ylim([0 1]);

        % Legend as plain text (no underscores as subscripts)
        legend('Location', 'southwest', 'Interpreter', 'none');
        hold off;
    end
end
