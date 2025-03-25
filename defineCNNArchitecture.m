function layers = defineCNNArchitecture(numClasses)
    layers = [
        imageInputLayer([40 100 1]) % Input layer [numBands, numFrames, channels]

        convolution2dLayer([20 9], 54, 'Stride', [1 1], 'Padding', 'same') % First convolutional layer
        reluLayer() % ReLU activation
        maxPooling2dLayer([1 3], 'Stride', [1 3]) % Max pooling in frequency

        convolution2dLayer([6 4], 54, 'Stride', [1 1], 'Padding', 'same') % Second convolutional layer
        reluLayer()

        fullyConnectedLayer(128)
        reluLayer()

        fullyConnectedLayer(numClasses) % Output layer (number of unique classes)
        softmaxLayer()
        classificationLayer()];
end