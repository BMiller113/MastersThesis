function [macs, params] = getModelMACs_CNNOnly(archType, freqBins, timeFrames, numClasses)
% getModelMACs_CNNOnly
% Computes CNN-only MACs (no preprocessing) for a single forward pass.
%
% Examples of usage:
% macs = getModelMACs_CNNOnly('trad-fpool3', 40, 32)
% macs = getModelMACs_CNNOnly('tpool2', 48, 32)
%
    if nargin < 4 || isempty(numClasses)
        numClasses = 12;  
    end

    archType = char(string(archType));

    layers = defineCNNArchitecture(numClasses, archType, freqBins, timeFrames);

    [macs, params] = estimateMACsAndParams(layers, [freqBins timeFrames 1]);
    
    % Console Output
    fprintf('\n=== CNN-only cost ===\n');
    fprintf('Arch: %s\n', archType);
    fprintf('Input: %dx%dx1\n', freqBins, timeFrames);
    fprintf('Params: %.3f K\n', params/1e3);
    fprintf('MACs:   %.3f M\n', macs/1e6);
    fprintf('Raw MACs: %.0f\n', macs);
end


function [macs, params] = estimateMACsAndParams(layers, inputSize)
% inputSize = [H W C]

    H = inputSize(1);
    W = inputSize(2);
    C = inputSize(3);

    macs   = 0;
    params = 0;

    for i = 1:numel(layers)
        L = layers(i);

        if isa(L,'nnet.cnn.layer.ImageInputLayer')
            continue;

        elseif isa(L,'nnet.cnn.layer.Convolution2DLayer')
            kH = L.FilterSize(1);
            kW = L.FilterSize(2);
            Cout = L.NumFilters;
            sH = L.Stride(1);
            sW = L.Stride(2);

            [Hout, Wout] = convOutSize(H, W, kH, kW, sH, sW, L.PaddingMode);

            % MACs
            macs = macs + double(Hout)*double(Wout)*double(Cout)*double(kH*kW*C);

            % Params (weights + bias)
            params = params + double(kH*kW*C*Cout);  % weights
            params = params + double(Cout);           % bias

            H = Hout; W = Wout; C = Cout;

        elseif isa(L,'nnet.cnn.layer.BatchNormalizationLayer')
            % gamma + beta per channel
            params = params + 2*double(C);

        elseif isa(L,'nnet.cnn.layer.FullyConnectedLayer')
            out = L.OutputSize;
            in  = double(H)*double(W)*double(C);

            macs   = macs + in * double(out);
            params = params + in * double(out);  % weights
            params = params + double(out);       % bias

            H = 1; W = 1; C = out;

        elseif isa(L,'nnet.cnn.layer.MaxPooling2DLayer') || ...
               isa(L,'nnet.cnn.layer.AveragePooling2DLayer')
            sH = L.Stride(1);
            sW = L.Stride(2);
            H = ceil(H / sH);
            W = ceil(W / sW);

        elseif isa(L,'nnet.cnn.layer.GlobalAveragePooling2DLayer')
            H = 1; W = 1;

        else
            continue;
        end
    end
end

function [Hout, Wout] = convOutSize(H, W, kH, kW, sH, sW, padMode)
% Output size helper

    if strcmpi(padMode,'same')
        Hout = ceil(H / sH);
        Wout = ceil(W / sW);
    else
        Hout = floor((H - kH) / sH) + 1;
        Wout = floor((W - kW) / sW) + 1;
    end

    Hout = max(1, Hout);
    Wout = max(1, Wout);
end
