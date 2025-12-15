function gpu_throughput_check
% gpu_throughput_check
% Quick GPU sanity/throughput test for a 40x32x1 CNN (tpool2-ish).
% Reports approximate images/sec for several batch sizes.

    % ----- Show selected GPU -----
    d = gpuDevice();  % NOTE: parentheses required
    fprintf('\nGPU: %s | CC %s | Mem %.2f GB (%.2f GB free)\n', ...
        d.Name, d.ComputeCapability, d.TotalMemory/1e9, d.AvailableMemory/1e9);

    % ----- Tiny tpool2-ish network (40x32x1 -> 14 classes) -----
    layers = [
        imageInputLayer([40 32 1],"Normalization","none")
        convolution2dLayer(3,32,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)   % time pooling vibe
        convolution2dLayer(3,64,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)
        fullyConnectedLayer(14)
        softmaxLayer
        classificationLayer];

    % ----- Try a few batch sizes -----
    batchSizes = [64 128 256 384 512];
    nIters = 150;  % forward passes to time per batch size

    % Warm-up once at a moderate batch size to JIT/cuDNN autotune
    warmBS = 256;
    Xw = randn(40,32,1,warmBS,'single');        % keep on CPU; MATLAB moves to GPU for training/predict
    Yw = categorical(randi(14,[warmBS,1]));
    optsWarm = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu', ...
        'MaxEpochs',1, 'MiniBatchSize', warmBS, ...
        'Shuffle','never','Verbose',false,'Plots','none');
    net = trainNetwork(Xw, Yw, layers, optsWarm);  % 1-epoch warm-up

    fprintf('Warm-up complete. Timing forward passes...\n');

    % Loop over batch sizes
    for bs = batchSizes
        X = randn(40,32,1,bs,'single');                  % CPU array is fine
        % Time pure predict() on GPU
        % (pre-call once to ensure full warm-up at this BS)
        predict(net, X, 'ExecutionEnvironment','gpu');
        g = gpuDevice(); reset(g); % clear any lingering kernels; optional

        tStart = tic;
        for i = 1:nIters
            predict(net, X, 'ExecutionEnvironment','gpu');
        end
        secs = toc(tStart);
        ips  = (bs * nIters) / secs;
        fprintf('Batch %4d: ~%.0f images/s (%.3fs over %d iters)\n', bs, ips, secs, nIters);
    end

    fprintf('\nPick the fastest batch size that fits in memory without OOM.\n');
    fprintf('If everything is very slow (<~1–2k img/s on a 4070 Ti), your runs are likely CPU-bound.\n');
end
