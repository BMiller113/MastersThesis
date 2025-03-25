% Load the dataset
[audioFiles, labels, testingFiles, validationFiles] = loadAudioData();

% Display dataset statistics
disp('=== Dataset Summary ===');
disp(['Training files: ', num2str(length(audioFiles))]);
disp(['Testing files: ', num2str(length(testingFiles))]);
disp(['Validation files: ', num2str(length(validationFiles))]);
numClasses = numel(categories(labels));
disp(['Number of classes: ', num2str(numClasses)]);
disp('=======================');

% Extract features
disp('Extracting features...');
trainingFeatures = extractFeatures(audioFiles);
testingFeatures = extractFeatures(testingFiles);
validationFeatures = extractFeatures(validationFiles);

% Define and train CNN
disp('Defining CNN architecture...');
layers = defineCNNArchitecture(numClasses);

disp('Training CNN...');
net = trainCNN(trainingFeatures, labels, layers);

% Evaluate model
disp('Evaluating model...');
accuracy = evaluateModel(net, testingFeatures, labels);

% Final results display
disp(' ');
disp('=== Final Results ===');
disp(['Test Accuracy: ', num2str(accuracy, '%.2f'), '%']);
disp('===================');