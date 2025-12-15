function measure_extractor_throughput(fileList, cfg, N)
    if nargin < 3, N = min(2000, numel(fileList)); end
    files = fileList(1:N);
    t = tic;
    X = extractFeatures(files, 'all', 'default', cfg); %#ok<NASGU>
    secs = toc(t);
    fprintf('extractFeatures: %d files in %.1fs  (%.1f files/s)\n', N, secs, N/secs);
end
