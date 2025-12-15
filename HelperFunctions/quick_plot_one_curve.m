function quick_plot_one_curve()
% QUICK sanity check: plot a single Sainath-style curve directly
% without any of the fancier overlay logic.
%
% Adjust 'fname' if you want to view a different model or condition.

    % Example: clean mel-only_default
    fname = fullfile(pwd, 'Results', 'sainath', 'clean', 'curves', ...
        'sainath_curve_events_transFA100ms_mel-only_default.csv');

    if ~isfile(fname)
        error('quick_plot_one_curve: missing file:\n  %s', fname);
    end

    T = readtable(fname);

    requiredCols = {'FA_per_hr','FR_fraction'};
    if ~all(ismember(requiredCols, T.Properties.VariableNames))
        error('quick_plot_one_curve: file %s missing required columns FA_per_hr / FR_fraction.', fname);
    end

    x = T.FA_per_hr;
    y = T.FR_fraction;

    mask = isfinite(x) & isfinite(y);
    x = x(mask);
    y = y(mask);

    if isempty(x)
        error('quick_plot_one_curve: no finite data in %s', fname);
    end

    [xSort, idx] = sort(x);
    ySort = y(idx);

    figure;
    plot(xSort, ySort, '-o', 'LineWidth', 1.5);
    grid on;

    maxX = max(xSort);
    maxY = max(ySort);

    % Autoscale but start at 0
    xlim([0, maxX * 1.05]);
    ylim([0, maxY * 1.05]);

    xlabel('False alarms per hour');
    ylabel('False reject rate (miss fraction)');
    title(sprintf('Sanity curve: %s', fname), 'Interpreter', 'none');
end
