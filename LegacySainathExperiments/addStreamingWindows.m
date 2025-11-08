function streams = addStreamingWindows(streams, cfg)
% Add sliding-window definitions and labels to each stream:
%   .winTimesMs : [N x 2] [start_ms end_ms] for each window
%   .winLabels  : [N x 1] categorical
    if nargin < 2, cfg = struct(); end
    spanMs = getfield_def(cfg,'streaming','winSpanMs', 500);
    hopMs  = getfield_def(cfg,'streaming','hopWinMs',  100);
    tolMs  = getfield_def(cfg,'sainath','labelTolMs',  100);

    % target words
    targets = {};
    if isfield(cfg,'warden') && isfield(cfg.warden,'targetWords') && ~isempty(cfg.warden.targetWords)
        targets = lower(string(cfg.warden.targetWords(:)'));
    end

    for s = 1:numel(streams)
        if ~isfield(streams(s),'wavPath') || ~isfield(streams(s),'events')
            error('Stream %d missing wavPath / events.', s);
        end
        info = audioinfo(streams(s).wavPath);
        Tms  = info.Duration * 1000;  % stream length in ms

        % Build sliding windows
        starts = 0:hopMs:max(0, Tms - spanMs);
        ends   = starts + spanMs;
        winTimesMs = [starts(:) ends(:)];

        % Label each window: keyword label if any event overlaps, else '_neg_'
        labs = strings(numel(starts),1);
        E = streams(s).events;
        if ~isempty(E)
            % prepare event intervals with tolerance in ms
            eStart = E.onset_s * 1000 - tolMs;
            eEnd   = E.offset_s * 1000 + tolMs;
            eLab   = lower(string(E.label));
            eStart = max(eStart, 0);
            eEnd   = max(eEnd, eStart);

            for i = 1:numel(starts)
                a = starts(i); b = ends(i);
                % find overlap
                ov = (eStart < b) & (eEnd > a);
                if any(ov)
                    if ~isempty(targets)
                        % pick the first event whose label is in targets; else fall back to the first overlap
                        idx = find(ov & ismember(eLab, targets), 1, 'first');
                        if isempty(idx), idx = find(ov,1,'first'); end
                    else
                        idx = find(ov,1,'first');
                    end
                    labs(i) = eLab(idx);
                else
                    labs(i) = "_neg_";
                end
            end
        else
            labs(:) = "_neg_";
        end

        % Commit into stream
        streams(s).winTimesMs = winTimesMs;
        streams(s).winLabels  = categorical(labs);
    end
end

function v = getfield_def(S, group, name, def)
    v = def;
    if ~isstruct(S), return; end
    if isfield(S, group)
        g = S.(group);
        if isfield(g, name) && ~isempty(g.(name)), v = g.(name); end
    end
end
