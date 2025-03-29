function layers = defineCNNArchitecture(numClasses, architectureType)
    % Define input size parameters
    freqBins = 40;   % Number of frequency bins
    timeSteps = 32;  % Number of time frames
    channels = 1;    % Mono channel
    inputSize = [freqBins timeSteps channels];
    
    switch architectureType
        case 'trad-fpool3'
            layers = [
                imageInputLayer(inputSize)
                
                convolution2dLayer([9 9], 64, 'Padding', 'same')
                batchNormalizationLayer()
                reluLayer()
                maxPooling2dLayer([3 3], 'Stride', [2 2])
                
                convolution2dLayer([5 5], 128, 'Padding', 'same')
                batchNormalizationLayer()
                reluLayer()
                maxPooling2dLayer([3 3], 'Stride', [2 2])
                
                dropoutLayer(0.5)
                fullyConnectedLayer(256)
                reluLayer()
                
                fullyConnectedLayer(numClasses)
                softmaxLayer()
                classificationLayer()
            ];
            
        case 'one-fstride4'
            layers = [
                imageInputLayer(inputSize)
                
                convolution2dLayer([8 8], 186, 'Stride', [1 4], 'Padding', 'same')
                batchNormalizationLayer()
                reluLayer()
                
                dropoutLayer(0.5)
                fullyConnectedLayer(256)
                reluLayer()
                
                fullyConnectedLayer(numClasses)
                softmaxLayer()
                classificationLayer()
            ];
            
        case 'tpool2'
            layers = [
                imageInputLayer(inputSize)
                
                convolution2dLayer([8 8], 94, 'Padding', 'same')
                batchNormalizationLayer()
                reluLayer()
                maxPooling2dLayer([3 3], 'Stride', [2 2])
                
                convolution2dLayer([5 5], 94, 'Padding', 'same')
                batchNormalizationLayer()
                reluLayer()
                
                dropoutLayer(0.5)
                fullyConnectedLayer(numClasses)
                softmaxLayer()
                classificationLayer()
            ];
            
        otherwise
            error('Unknown architecture type: %s', architectureType);
    end
end