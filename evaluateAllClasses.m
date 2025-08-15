function [perClassTbl, macro] = evaluateAllClasses(net, XTest, YTest, makePlots)
    if nargin < 4, makePlots = false; end
    if ~iscategorical(YTest), YTest = categorical(YTest); end

    C = categories(YTest);
    rows = cell(numel(C), 6);  % Nx6 matrix, not Nx1 of cells

    for i = 1:numel(C)
        pos = C{i};
        try
            [~, FR, FA, rocInfo] = evaluateModel(net, XTest, YTest, false, pos);
            AUC = rocInfo.AUC;
            thr = rocInfo.thrUsed;
        catch ME
            % NaNs should keep table shape valid in bad cases
            warning('Per-class eval failed for "%s": %s', pos, ME.message);
            AUC = NaN; thr = NaN; FR = NaN; FA = NaN;
        end
        rows(i,:) = {pos, AUC, thr, FR, FA, sum(YTest==pos)};
    end

    perClassTbl = cell2table(rows, ...
        'VariableNames', {'Class','AUC','Thr','FR','FA','Support'});

    macro.FR  = mean(perClassTbl.FR,  'omitnan');
    macro.FA  = mean(perClassTbl.FA,  'omitnan');
    macro.AUC = mean(perClassTbl.AUC, 'omitnan');

    if makePlots
        figure('Visible','on');
        histogram(perClassTbl.AUC);
        title('Per-class AUC distribution'); xlabel('AUC'); ylabel('Count'); grid on;
    end
end