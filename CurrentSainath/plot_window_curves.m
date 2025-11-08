function plot_window_curves(curveDir, plotTitle, xMax)
% plot_window_curves  overlay FR vs FA/h from our CSVs
% usage:
%   plot_window_curves(dir);                     % auto title, auto x
%   plot_window_curves(dir, 'CLEAN');            % auto x
%   plot_window_curves(dir, 'CLEAN', 2000);      % forced x=0..2000
%
    if nargin < 1 || isempty(curveDir)
        curveDir = fullfile(pwd, 'Results', 'sainath', 'clean', 'curves');
    end
    if nargin < 2 || isempty(plotTitle)
        plotTitle = 'Window-level FR vs FA/h';
    end
    if nargin < 3 || isempty(xMax)
        xMax = [];  % let MATLAB autoscale
    end

    d = dir(fullfile(curveDir, 'sainath_curve_events_*.csv'));
    if isempty(d)
        error('No CSVs in %s', curveDir);
    end

    figure; hold on; grid on;
    for k = 1:numel(d)
        T = readtable(fullfile(d(k).folder, d(k).name));
        [~,bn,~] = fileparts(d(k).name);
        tag = erase(bn, 'sainath_curve_events_');
        plot(T.FA_per_hr, T.FR_fraction, 'LineWidth', 2, 'DisplayName', tag);
    end
    xlabel('False Alarms per hour');
    ylabel('False Rejects (fraction)');
    title(plotTitle, 'Interpreter','none');
    lg = legend('Location','best');
    set(lg,'Interpreter','none');  % no subscripts

    % force same x-axis for all plots if requested
    if ~isempty(xMax)
        xlim([0 xMax]);
    end
    ylim([0 1]);
end
