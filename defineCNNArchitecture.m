function layers = defineCNNArchitecture(numClasses, architectureType, freqBins, timeSteps)
% defineCNNArchitecture: build a CNN with variable input size.
% 
% Examples:
% layers = defineCNNArchitecture(numClasses, archType, freqBins, timeSteps)
% layers = defineCNNArchitecture(numClasses, archType)  % defaults to 40x32

    % Defaults for older uses (check to see if still necessary !!! 8/15)
    if nargin < 3 || isempty(freqBins),  freqBins  = 40; end
    if nargin < 4 || isempty(timeSteps), timeSteps = 32; end

    inputSize = [freqBins, timeSteps, 1];

    switch lower(architectureType)

        case 'trad-fpool3'
            layers = [ ...
                imageInputLayer(inputSize, "Normalization","none"); 
                
                convolution2dLayer([9 9], 64, "Padding","same");
                batchNormalizationLayer;
                reluLayer;
                maxPooling2dLayer([3 3], "Stride",[2 2]);

                convolution2dLayer([5 5], 128, "Padding","same");
                batchNormalizationLayer;
                reluLayer;
                maxPooling2dLayer([3 3], "Stride",[2 2]);

                dropoutLayer(0.5);
                fullyConnectedLayer(256);
                reluLayer;

                fullyConnectedLayer(numClasses);
                softmaxLayer;
                classificationLayer ];

        case 'one-fstride4'
            % First conv spans a wide time window; keep padding same
            kH = 9;
            kW = min(timeSteps, max(3, round(timeSteps)));
            layers = [ ...
                imageInputLayer(inputSize, "Normalization","none");

                convolution2dLayer([kH kW], 128, "Stride",[1 4], "Padding","same");
                batchNormalizationLayer;
                reluLayer;

                convolution2dLayer([5 1], 128, "Padding","same");
                batchNormalizationLayer;
                reluLayer;

                globalAveragePooling2dLayer;

                dropoutLayer(0.5);
                fullyConnectedLayer(256);
                reluLayer;

                fullyConnectedLayer(numClasses);
                softmaxLayer;
                classificationLayer ];

        case 'tpool2'
            layers = [ ...
                imageInputLayer(inputSize, "Normalization","none");

                convolution2dLayer([8 8], 94, "Padding","same");
                batchNormalizationLayer;
                reluLayer;
                maxPooling2dLayer([3 3], "Stride",[2 2]);

                convolution2dLayer([5 5], 94, "Padding","same");
                batchNormalizationLayer;
                reluLayer;

                dropoutLayer(0.5);
                fullyConnectedLayer(numClasses);
                softmaxLayer;
                classificationLayer ];

        otherwise
            error('Unknown architecture type: %s', architectureType);
    end
end
