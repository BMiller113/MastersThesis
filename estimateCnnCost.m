function cost = estimateCnnCost(layers, inputSize)
% estimateCnnCost: approximate parameter count + MACs for conv/fc layers.
% inputSize: [H W C]

H = inputSize(1); W = inputSize(2); C = inputSize(3);

params = 0;
macs   = 0;

for i = 1:numel(layers)
    L = layers(i);

    if isa(L,'nnet.cnn.layer.Convolution2DLayer')
        kH = L.FilterSize(1); kW = L.FilterSize(2);
        F  = L.NumFilters;
        % params: (kH*kW*C)*F + bias(F)
        params = params + (kH*kW*C)*F + F;

        % output spatial size approx (same padding keeps H,W; stride reduces)
        sH = L.Stride(1); sW = L.Stride(2);
        outH = ceil(H / sH);
        outW = ceil(W / sW);

        % MACs: outH*outW * F * (kH*kW*C)
        macs = macs + outH*outW * F * (kH*kW*C);

        % update channels; update H,W
        C = F; H = outH; W = outW;

    elseif isa(L,'nnet.cnn.layer.FullyConnectedLayer')
        F = L.OutputSize;
        % Flatten size is H*W*C
        inN = H*W*C;
        params = params + inN*F + F;
        macs   = macs   + inN*F;
        H = 1; W = 1; C = F;

    elseif isa(L,'nnet.cnn.layer.MaxPooling2DLayer')
        sH = L.Stride(1); sW = L.Stride(2);
        H = ceil(H / sH);
        W = ceil(W / sW);

    elseif isa(L,'nnet.cnn.layer.GlobalAveragePooling2DLayer')
        H = 1; W = 1;
    end
end

cost = struct();
cost.Params = params;
cost.MACs   = macs;
cost.InputSize = inputSize;
end
