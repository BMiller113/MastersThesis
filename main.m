% Main script for CNN-based Keyword Spotting
% using Google Speech Commands V2

%% 1. Load/Prepare Data                                             %% for section break
[audioFiles, labels, testingFiles, testingLabels] = loadAudioData();

% Convert labels to categorical arrays
labels = categorical(labels);
testingLabels = categorical(testingLabels);

% Verify label conversion
disp('=== Label Verification ===');
disp(['Training labels type: ', class(labels)]);
disp(['Test labels type: ', class(testingLabels)]);
disp(['Number of classes: ', num2str(numel(categories(labels)))]);
disp('Unique categories:');
disp(unique(labels))
disp('Class distribution:'); 
tabulate(labels)
disp('=======================');

%% 2. Feature Extraction
disp('Extracting features...');
trainingFeatures = extractFeatures(audioFiles);
testingFeatures = extractFeatures(testingFiles);

% Feature sanity checks
disp('=== Feature Validation ===');
disp(['Training features size: ', mat2str(size(trainingFeatures))]);
disp(['Testing features size: ', mat2str(size(testingFeatures))]);
disp(['NaN features in training: ', num2str(sum(isnan(trainingFeatures(:))))]);
disp(['Inf features in training: ', num2str(sum(isinf(trainingFeatures(:))))]);
disp(['Feature range: [', num2str(min(trainingFeatures(:))), ', ', num2str(max(trainingFeatures(:))), ']']);
disp('=======================');

% Visualize sample spectrogram
figure;
imshow(squeeze(trainingFeatures(:,:,1,1)), []);
title('Sample Training Spectrogram');
colorbar;

%% 3. Model Configuration/Select Architecture
archType = 'one-fstride4'; % Options: 'trad-fpool3', 'one-fstride4', 'tpool2'
disp(['Selected architecture: ', archType]);

% Define CNN architecture
disp('Defining CNN architecture...');
layers = defineCNNArchitecture(numel(categories(labels)), archType);
disp(layers);

%% 4. Training
disp('Training CNN...');
net = trainCNN(trainingFeatures, labels, layers);

%% 5. Evaluation
% Test set verification
disp('=== Test Set Verification ===');
disp(['Test samples: ', num2str(numel(testingLabels))]);
disp(['Test features shape: ', mat2str(size(testingFeatures))]);
disp('First 5 test labels:');
disp(testingLabels(1:min(5,end)))
disp('Test set class distribution:');
tabulate(testingLabels)
disp('=======================');

% Evaluate model
disp('Evaluating model...');
[accuracy, FR, FA] = evaluateModel(net, testingFeatures, testingLabels);

%% 6. Final
disp('=== Final Results ===');
disp(['Test Accuracy: ', num2str(accuracy, '%.2f'), '%']);
disp(['False Reject Rate: ', num2str(FR, '%.2f'), '%']);
disp(['False Alarm Rate: ', num2str(FA, '%.2f'), '%']);
disp('===================');

% Save model
save('trained_kws_model.mat', 'net', 'archType');