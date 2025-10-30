function streams = addStreamingWindows(streams, cfg)
% Add sliding-window definitions and labels to each stream (PRE-HANGOVER STYLE):
%   .winTimesMs : [N x 2] [start_ms end_ms] for each decision window
%   .winLabels  : [N x 1] categorical (keyword label if overlaps, else '_neg_')

    if nargin < 2, cfg = struct(); end

    % --- Geometry: Sainath (25 ms window, 10 ms hop, 23L+1+8R = 32 frames) ---
    frameMs = getf(cfg,'features','frameMs', 25);
    hopMs   = getf(cfg,'features','hopMs',   10);
    L       = getf(cfg,'sainath','leftCtx',  23);
    R       = getf(cfg,'sainath','rightCtx', 8);

    % Decision window span ≈ width of the stacked context (pre-hangover code)
    % For 25/10 & 23L+8R this is 25 + 31*10 = 335 ms
    spanMs  = getf(cfg,'streaming','winSpanMs', frameMs + hopMs*(L+R));
    hopWin  = getf(cfg,'streaming','hopWinMs', 10);   % decision every 10 ms (critical!)
    tolMs   = getf(cfg,'sainath','labelTolMs', 200);  % generous tol to guarantee positives

    % Target keyword set (lowercase)
    targets = string([]);
    if isfield(cfg,'sainath') && isfield(cfg.sainath,'targetWords') && ~isempty(cfg.sainath.targetWords)
        targets = lower(string(cfg.sainath.targetWords(:)'));
    elseif isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
        targets = lower(string(cfg.warden.targetWords(:)'));
    end
    targets = targets(:);

    for s = 1:numel(streams)
        assert(isfield(streams(s),'wavPath') && isfield(streams(s),'events'), ...
               'Stream %d missing wavPath/events', s);

        info = audioinfo(streams(s).wavPath);
        Tms  = round(info.Duration * 1000);

        if Tms <= 0
            streams(s).winTimesMs = [0 spanMs];
            streams(s).winLabels  = categorical("_neg_");
            continue;
        end

        % Decision window centers, evenly spaced by hopWin
        halfWin = spanMs/2;
        centers = halfWin:hopWin:max(halfWin, Tms - halfWin);
        if isempty(centers), centers = halfWin; end

        starts = centers - halfWin;
        ends   = centers + halfWin;
        winTimesMs = [starts(:) ends(:)];

        % Label by overlap (with tolerance padding)
        labs = strings(numel(centers),1);
        E = streams(s).events;
        if ~isempty(E)
            evOn = 1000*E.onset_s(:)  - tolMs;
            evOff= 1000*E.offset_s(:) + tolMs;
            evOn  = max(evOn, 0);
            evOff = max(evOff, evOn);

            evLab = lower(string(E.label(:)));
            for i = 1:numel(centers)
                a = starts(i); b = ends(i);
                ov = (evOn < b) & (evOff > a);
                if any(ov)
                    if ~isempty(targets)
                        % prefer a target word if any overlaps
                        idx = find(ov & ismember(evLab, targets), 1, 'first');
                        if isempty(idx), idx = find(ov,1,'first'); end
                    else
                        idx = find(ov,1,'first');
                    end
                    labs(i) = evLab(idx);
                else
                    labs(i) = "_neg_";
                end
            end
        else
            labs(:) = "_neg_";
        end

        streams(s).winTimesMs = winTimesMs;
        streams(s).winLabels  = categorical(labs);
    end
end

% ---- local getter ----
function v = getf(S, group, name, def)
    v = def;
    if ~isstruct(S) || ~isfield(S, group), return; end
    G = S.(group);
    if isfield(G, name) && ~isempty(G.(name)), v = G.(name); end
end
