function plotOverlayAndSaveROC(rocInfos, labels, overlayKey, cfg)
% plotOverlayAndSaveROC
% Overlays multiple curves from rocInfo structs (with fields far,frr).
% style='roc' -> plots TPR vs FPR (true ROC)
% style='det' -> plots FRR vs FPR (DET-like error plot)

    if nargin < 4 || isempty(cfg), cfg = kws_config(); end
    if isempty(rocInfos), return; end
    if nargin < 2 || isempty(labels), labels = repmat({''}, size(rocInfos)); end

    style = 'roc';
    if isfield(cfg,'plots') && isfield(cfg.plots,'overlay') && isfield(cfg.plots.overlay,'style') && ~isempty(cfg.plots.overlay.style)
        style = lower(string(cfg.plots.overlay.style));
    elseif isfield(cfg,'plots') && isfield(cfg.plots,'roc') && isfield(cfg.plots.roc,'style') && ~isempty(cfg.plots.roc.style)
        % fallback to main roc style if overlay style not specified
        style = lower(string(cfg.plots.roc.style));
    end

    % Output directory
    outDir = cfg.paths.outputDir;
    sub = fullfile('roc','overlay');
    if isfield(cfg.plots,'overlay') && isfield(cfg.plots.overlay,'outSubdir') && ~isempty(cfg.plots.overlay.outSubdir)
        sub = cfg.plots.overlay.outSubdir;
    end
    outDir = fullfile(outDir, sub);
    if ~exist(outDir,'dir'), mkdir(outDir); end

    % Build plot
    fig = figure('Visible','on'); hold on; grid on;

    anyPlotted = false;
    for i = 1:numel(rocInfos)
        rc = rocInfos{i};
        if ~isstruct(rc) || ~isfield(rc,'far') || ~isfield(rc,'frr'), continue; end
        far = rc.far(:);
        frr = rc.frr(:);
        if numel(far) < 2 || numel(frr) < 2 || numel(far) ~= numel(frr), continue; end

        x = far * 100; % FPR in %
        if style == "roc"
            y = (1 - frr); % TPR
        else
            y = frr * 100; % FRR in %
        end

        plot(x, y, 'LineWidth', 2);
        anyPlotted = true;
    end

    if ~anyPlotted
        title('Overlay ROC/DET — no valid curves'); axis off;
        return;
    end

    xlabel('False Positive Rate (%)');
    if style == "roc"
        ylabel('True Positive Rate');
        title(sprintf('ROC Overlay — %s', overlayKey), 'Interpreter','none');
        ylim([0 1]);
    else
        ylabel('False Reject Rate (%)');
        title(sprintf('DET (FPR vs FRR) Overlay — %s', overlayKey), 'Interpreter','none');
    end

    % X-axis zoom
    xlimDefault = [0 5];
    if isfield(cfg.plots,'overlay') && isfield(cfg.plots.overlay,'xlimPercent') && ~isempty(cfg.plots.overlay.xlimPercent)
        xlim(cfg.plots.overlay.xlimPercent);
    else
        xlim(xlimDefault);
    end

    % Legend
    try
        legend(labels, 'Interpreter','none', 'Location','best');
    catch
        % ignore legend failures
    end

    % Save
    saveIt = false;
    if isfield(cfg.plots,'overlay') && isfield(cfg.plots.overlay,'saveToDisk')
        saveIt = cfg.plots.overlay.saveToDisk;
    end
    if saveIt
        try, saveas(fig, fullfile(outDir, sprintf('%s.png', overlayKey))); catch, end
        try, savefig(fig, fullfile(outDir, sprintf('%s.fig', overlayKey))); catch, end
    end
end
