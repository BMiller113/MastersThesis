function S = slidingWindowScores(net, wavPath, cfg)
% slidingWindowScores (auto-matching)
% Builds stacks that MATCH the network's input geometry automatically.
% Works for nets expecting [B x W x 1], e.g., [48 x 32 x 1] or [40 x 32 x 1].
%
% Returns:
%   S.scores   [Nwin x C]
%   S.t_center [Nwin x 1] (sec)
%   S.classes  {1xC}

% ----- 1) Infer input geometry from the net -----
inSz = [];
try
    inSz = net.Layers(1).InputSize;
catch
    try, inSz = net.InputSize; end
end
assert(~isempty(inSz) && numel(inSz) >= 2, ...
    'Could not read net input size; expected [bands x frames x channels].');

bandsNet = inSz(1);      % e.g., 48 (or 40)
framesNet = inSz(2);     % e.g., 32
chNet = (numel(inSz) >= 3) * inSz(min(3,numel(inSz)));  % tolerate 2-D specs
if chNet == 0, chNet = 1; end
assert(chNet == 1, 'Expected single-channel input; got %d.', chNet);

% ----- 2) Map Sainath stacking to this width -----
% Target frames = left + 1 + right. Keep rightCtx<=8 if possible; fill left.
rightCtxDefault = 8;
if isfield(cfg,'sainath') && isfield(cfg.sainath,'rightCtx') && ~isempty(cfg.sainath.rightCtx)
    rightCtxDefault = cfg.sainath.rightCtx;
end
rightCtx = min(rightCtxDefault, max(0, framesNet-1));   % cap to width-1
leftCtx  = max(0, framesNet-1 - rightCtx);

% Persist back to cfg so downstream code is consistent
try
    cfg.sainath.rightCtx = rightCtx;
    cfg.sainath.leftCtx  = leftCtx;
catch, end

% Frame/hop
frameMs = getf(cfg,'features','frameMs', 25);
hopMs   = getf(cfg,'features','hopMs',   10);

% ----- 3) Load audio -----
[x, fs] = audioread(wavPath);
if size(x,2) > 1, x = mean(x,2); end
if fs ~= 16000, x = resample(x,16000,fs); fs = 16000; end
x = single(max(-1,min(1,x)));

% ----- 4) Full log-mel at the REQUIRED # bands -----
[Mlog, t_ms] = fullLogMelMatrix(x, fs, bandsNet, frameMs, hopMs);

% ----- 5) Build decision centers to fill the stream with overlapping windows -----
Tms = t_ms(end) + frameMs/2;
spanMs  = frameMs + hopMs*(leftCtx + rightCtx);
halfWin = spanMs/2;
hopWin  = getf(cfg,'streaming','hopWinMs', hopMs);
centers = (halfWin):hopWin:(Tms - halfWin);
if isempty(centers), centers = halfWin; end
Nw = numel(centers);

% Map each center to the nearest frame index
idxFrames = round((centers(:) - t_ms(1)) / hopMs) + 1;
idxFrames = max(1, min(idxFrames, size(Mlog,2)));

% ----- 6) Stack [bandsNet x framesNet x 1 x Nw] with edge padding -----
X4 = zeros(bandsNet, framesNet, 1, Nw, 'single');
for j = 1:Nw
    c = idxFrames(j);
    left  = max(1, c - leftCtx);
    right = min(size(Mlog,2), c + rightCtx);

    block = Mlog(:, left:right);
    % pad to framesNet (edge-repeat)
    curW = size(block,2);
    if curW < framesNet
        need = framesNet - curW;
        padLeft  = max(0, leftCtx  - (c-left));
        padRight = need - padLeft;
        if padLeft > 0,  block = [repmat(Mlog(:,left),  1, padLeft),  block]; end
        if padRight > 0, block = [block, repmat(Mlog(:,right), 1, padRight)]; end
    end
    X4(:,:,1,j) = block(:,1:framesNet);
end

% Z-score per stream (simple)
mu = mean(X4(:)); sd = std(X4(:)) + eps;
X4z = (X4 - mu) / sd;

% ----- 7) Predict -----
scores = predict(net, X4z);   % [Nw x C]
% Ensure probabilities (softmax) if needed
if ~isempty(scores)
    rowSums = sum(scores,2,'omitnan');
    if any(~isfinite(rowSums)) || median(abs(rowSums - 1)) > 1e-3
        mx = max(scores, [], 2);
        ex = exp(scores - mx);
        scores = ex ./ max(eps, sum(ex,2));
    end
end

% ----- 8) Finalize -----
S = struct();
S.scores    = scores;
S.t_center  = centers(:)/1000;
S.classes   = tryGetClasses(net);
end

% ===== helpers =====
function [Mlog, t_ms] = fullLogMelMatrix(x, fs, numBands, frameMs, hopMs)
frameLen = round(fs * frameMs/1000);
hopSamp  = max(1, round(fs * hopMs /1000));
ovl      = max(0, frameLen - hopSamp);
win      = localHamming(frameLen);

if exist('melSpectrogram','file') == 2
    try
        M = melSpectrogram(x, fs, 'Window',win, 'OverlapLength',ovl, ...
                           'NumBands', numBands, 'FrequencyRange',[50 min(7000,fs/2*0.999)]);
        Mlog = log10(M + eps);
    catch
        M = melSpectrogram(x, fs, 'Window',win, 'OverlapLength',ovl, 'NumBands', numBands);
        Mlog = log10(M + eps);
    end
else
    S = spectrogram(x, win, ovl, numel(win), fs);
    fb = designAuditoryFilterBank(fs, 'NumBands',numBands, 'FFTLength',numel(win), ...
                                  'FrequencyRange',[50 min(7000,fs/2*0.999)]);
    Mlog = log10(fb * abs(S) + eps);
end

nF = size(Mlog,2);
t_centers = ((0:nF-1) * hopSamp + frameLen/2) / fs;  % seconds
t_ms = t_centers(:) * 1000;
end

function w = localHamming(N)
try, w = hamming(N,'periodic'); catch, w = hamming(N); end
end

function v = getf(cfg, group, name, def)
v = def;
if ~isstruct(cfg), return; end
if isfield(cfg, group)
    g = cfg.(group);
    if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
end
end

function cls = tryGetClasses(net)
    try, cls = cellstr(string(net.Layers(end).Classes));
    catch,   cls = {'unknown'};
    end
end
