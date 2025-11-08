function [FAh, FR, thr, AUC] = evaluate_windows_simple(net, streams, cfg)
% Window-level sweep using pKW = max over non-filler classes.
% FR = 1-TPR over window labels; FA/hour from 10ms decisions.
% This is the simple pre-hangover evaluator that should yield curved lines
% as long as positive/negative score distributions overlap.

% Collect scores S and labels Y (logical)
[S, Y] = collect_S_Y(net, streams, cfg);

% Threshold grid
lo = max(0, min(S) - 1e-6);
hi = min(1, max(S) + 1e-6);
thr = linspace(lo, hi, 2001).';

% Vectorized sweep
Srow = S(:)'; Yrow = Y(:)'; T = thr(:)';
predPos = Srow >= T.';   % K x N
TP = sum(predPos &  Yrow, 2);
FP = sum(predPos & ~Yrow, 2);
P  = max(1, sum(Y==1));
N  = max(1, sum(Y==0));
tpr = TP ./ P;
fpr = FP ./ N;
FR  = 1 - tpr;

% Sort by FPR and convert to FA/hour
[fpr, idx] = sort(fpr,'ascend'); FR = FR(idx); thr = thr(idx);
hopMs = getf(cfg,'features','hopMs',10);
FAh   = fpr * (3600/(hopMs/1000));

% AUC for reference
AUC = trapz(fpr, tpr(idx));

% De-dupe FA bins, enforce monotone FR (like Sainath)
[FAh, FR] = dedupe_monotone(FAh, FR);

end

% -------- helpers ----------
function [S, Y] = collect_S_Y(net, streams, cfg)
    fillers = ["_silence_","_unknown_","_background_noise_","_neg_"];
    B=40; frameMs=25; hopMs=10; L=23; R=8; W=L+1+R;
    allS = []; allY = [];
    for i=1:numel(streams)
        [x,fs] = audioread(streams(i).wavPath);
        if size(x,2)>1, x=mean(x,2); end
        if fs~=16000, x = resample(x,16000,fs); fs=16000; end
        [Mlog,tms] = fullLogMelMatrix(x,fs,B,frameMs,hopMs);
        c_ms = mean(streams(i).winTimesMs,2);
        idx = round((c_ms - tms(1))/hopMs) + 1;
        idx = max(1, min(idx, size(Mlog,2)));
        Nw = numel(idx);
        X4 = zeros(B,W,1,Nw,'single');
        for j=1:Nw
            c=idx(j); left=max(1,c-L); right=min(size(Mlog,2),c+R);
            block=Mlog(:,left:right);
            if size(block,2)<W
                need=W-size(block,2);
                padL=max(0,L-(c-left)); padR=need-padL;
                if padL>0, block=[repmat(Mlog(:,left),1,padL),block]; end
                if padR>0, block=[block,repmat(Mlog(:,right),1,padR)]; end
            end
            X4(:,:,1,j)=block(:,1:W);
        end
        mu=mean(X4(:)); sd=std(X4(:))+eps; X4=(X4-mu)/sd;
        P = predict(net,X4);                   % [Nw x C]
        cls = string(tryGetClasses(net));
        mask = ~startsWith(cls,"_");
        if ~any(mask), mask = true(1,size(P,2)); end
        pKW = max(P(:,mask),[],2);             % trigger score
        wlab = string(streams(i).winLabels(:));
        isPos = ~ismember(lower(wlab), lower(fillers));
        allS = [allS; pKW(:)]; %#ok<AGROW>
        allY = [allY; double(isPos(:))]; %#ok<AGROW>
    end
    S = allS; Y = allY>0;
end

function [Mlog, t_ms] = fullLogMelMatrix(x, fs, numBands, frameMs, hopMs)
    frameLen = round(fs*frameMs/1000);
    hopSamp  = max(1, round(fs*hopMs/1000));
    ovl      = max(0, frameLen-hopSamp);
    w = localHamming(frameLen);
    M = melSpectrogram(x,fs,'Window',w,'OverlapLength',ovl,'NumBands',numBands, ...
                       'FrequencyRange',[50 min(7000,fs/2*0.999)]);
    Mlog = log10(M+eps);
    nF = size(Mlog,2);
    t_cent = ((0:nF-1)*hopSamp + frameLen/2) / fs; % sec
    t_ms = t_cent(:)*1000;
end
function w = localHamming(N), try w=hamming(N,'periodic'); catch, w=hamming(N); end, end
function classes = tryGetClasses(net)
    classes = "unknown";
    try classes = string(net.Layers(end).Classes); catch, end
end
function [xU, yMono] = dedupe_monotone(x, y)
    [x, idx] = sort(x,'ascend'); y = y(idx);
    [xU,~,g] = unique(x,'stable');
    yU = accumarray(g, y, [], @min);
    yMono = flip(cummin(flip(yU)));  % enforce non-increasing FR
end
function v = getf(cfg, group, name, def)
    v = def;
    if isstruct(cfg) && isfield(cfg,group) && isfield(cfg.(group),name)
        tmp = cfg.(group).(name); if ~isempty(tmp), v = tmp; end
    end
end
