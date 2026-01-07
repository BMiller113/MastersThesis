function [net, modelPath, didTrain] = getOrTrainModel(tag, XTrain, YTrain, layers, cfg)
% getOrTrainModel: loads a cached model if present; otherwise trains and saves.
if nargin < 5 || isempty(cfg), cfg = kws_config(); end

dsTag = lower(string(cfg.dataset.version));
arch  = lower(string(cfg.model.arch));
B = cfg.features.baseBands;
W = cfg.features.targetFrames;
key = sprintf('%s__%s__%s__%dx%d', dsTag, arch, tag, B, W);

modelPath = fullfile(cfg.paths.modelDir, sprintf('model_%s.mat', key));

didTrain = true;
if cfg.cache.enableModelCache && exist(modelPath,'file')
    S = load(modelPath, 'net');
    if isfield(S,'net')
        net = S.net;
        didTrain = false;
        fprintf('Loaded cached model: %s\n', modelPath);
        return;
    end
end

net = trainCNN(XTrain, YTrain, layers, cfg);
save(modelPath, 'net', 'key', 'arch', 'dsTag', '-v7.3');
fprintf('Saved model: %s\n', modelPath);

end
