function plotAndSaveROC(rocInfo, runKey, cfg)
% plotAndSaveROC
% Uses rocInfo.far (FPR) and rocInfo.frr (FRR) from evaluateModel().
% If cfg.plots.roc.style == 'roc', plots TPR vs FPR (true ROC).
% If cfg.plots.roc.style == 'det', plots FRR vs FPR (DET-like error plot).
%

    if nargin < 3 || isempty(cfg), cfg = kws_config(); end
    if ~isstruct(rocInfo) || ~isfield(rocInfo,'far') || ~isfield(rocInfo,'frr')
        return;
    end

    far = rocInfo.far(:);
    frr = rocInfo.frr(:);
    if numel(far) < 2 || numel(frr) < 2 || numel(far) ~= numel(frr)
        return;
    end

    style = 'roc';
    if isfield(cfg,'plots') && isfield(cfg.plots,'roc') && isfield(cfg.plots.roc,'style') && ~isempty(cfg.plots.roc.style)
        style = lower(string(cfg.plots.roc.style));
    end

    % Build output directory
    outDir = cfg.paths.outputDir;
    sub = 'roc';
    if isfield(cfg.plots,'roc') && isfield(cfg.plots.roc,'outSubdir') && ~isempty(cfg.plots.roc.outSubdir)
        sub = cfg.plots.roc.outSubdir;
    end
    outDir = fullfile(outDir, sub);
    if ~exist(outDir,'dir'), mkdir(outDir); end

    % Compute Y axis based on style
    x = far * 100; % percent
    switch style
        case "roc"
            y = (1 - frr);          % TPR in [0,1]
            ylab = 'True Positive Rate';
            ttl  = sprintf('ROC — %s', runKey);
        otherwise
            y = frr * 100;          % FRR in %
            ylab = 'False Reject Rate (%)';
            ttl  = sprintf('DET (FPR vs FRR) — %s', runKey);
    end

    % Plot
    fig = figure('Visible','on'); grid on; hold on;
    plot(x, y, 'LineWidth', 2);

    xlabel('False Positive Rate (%)');
    ylabel(ylab);
    title(ttl, 'Interpreter','none');

    % Axis limits
    if isfield(cfg.plots,'roc') && isfield(cfg.plots.roc,'xlimPercent') && ~isempty(cfg.plots.roc.xlimPercent)
        xlim(cfg.plots.roc.xlimPercent);
    end
    if style == "roc"
        ylim([0 1]);
    else
        % keep auto unless you want a default zoom
    end

    % Save
    if isfield(cfg.plots,'roc') && isfield(cfg.plots.roc,'saveToDisk') && cfg.plots.roc.saveToDisk
        try, saveas(fig, fullfile(outDir, sprintf('%s.png', runKey))); catch, end
        try, savefig(fig, fullfile(outDir, sprintf('%s.fig', runKey))); catch, end
    end
end
