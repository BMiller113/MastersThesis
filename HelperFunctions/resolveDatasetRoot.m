function root = resolveDatasetRoot(cfg)
% resolveDatasetRoot: returns dataset root based on cfg.dataset.version
v = lower(string(cfg.dataset.version));
switch v
    case "v1"
        root = cfg.paths.datasetRootV1;
    case "v2"
        root = cfg.paths.datasetRootV2;
    otherwise
        error('Unknown cfg.dataset.version: %s (use "v1" or "v2")', v);
end
if ~exist(root,'dir')
    error('Dataset root not found: %s', root);
end
end
