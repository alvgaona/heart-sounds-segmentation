close all
clc

%% Dataset configuration and set up
cd('datasets/david_springer')
load('springer_dataset.mat')

Fs = 1000; % Hz
signals = springer_dataset.audio_data;
labels = springer_dataset.labels;
features = springer_dataset.features';

lengthSignals = cellfun(@(x) length(x), signals);

signalLen = 2000;

signals = signals(lengthSignals >= signalLen);
labels = labels(lengthSignals >= signalLen);

% Divide observations with K-Folds Cross Validation method (K = 10);
k = 10; 
groups = kFolds(signals, labels, k);

%% LSTM Configuration

MiniBatchSize = 50;

layers = [ ...
      sequenceInputLayer(44)
      lstmLayer(200,'OutputMode','sequence')
      fullyConnectedLayer(4)
      softmaxLayer
      classificationLayer
      ];

options = trainingOptions('adam', ...
  'MaxEpochs',10, ...
  'MiniBatchSize',MiniBatchSize, ...
  'InitialLearnRate',0.01, ...
  'LearnRateDropPeriod',3, ...
  'LearnRateSchedule','piecewise', ...
  'GradientThreshold',1, ...
  'Plots','training-progress',...
  'Verbose',1);

%% LSTM Input Configuration
n = 2000;
stride = 2000;

kFoldsNets = cell(1,k);

for i=3:3
  disp("Group " + i)

  signalsTrain = groups{i}.train(:,1);
  labelsTrain = groups{i}.train(:,2);
  signalsTest = groups{i}.test(:,1);
  labelsTest = groups{i}.test(:,2);

  [framedSignalsTrain, framedLabelsTrain] = frameSignals(signalsTrain, labelsTrain, stride, n);
  [framedSignalsTest, framedLabelsTest] = frameSignals(signalsTest, labelsTest, stride, n);

  %% FSST Feature extraction
  signalsFsstTrain = cell(size(framedSignalsTrain));
  meanTrain = cell(1,length(framedSignalsTrain));
  stdTrain = cell(1,length(framedSignalsTrain));

  disp("Feature extraction (training)")
  for idx = 1:length(framedSignalsTrain)
     [s,f,t] = fsst(framedSignalsTrain{idx},Fs,kaiser(128));

     f_indices = (f > 25) & (f < 200);
     signalsFsstTrain{idx}= [real(s(f_indices,:)); imag(s(f_indices,:))];

     meanTrain{idx} = mean(signalsFsstTrain{idx},2);
     stdTrain{idx} = std(signalsFsstTrain{idx},[],2);
  end

  standardizeFun = @(x) (x - mean(cell2mat(meanTrain),2))./mean(cell2mat(stdTrain),2);
  signalsFsstTrain = cellfun(standardizeFun,signalsFsstTrain,'UniformOutput',false);

  signalsFsstTest = cell(size(framedSignalsTest));

  disp("Feature extraction (evaluation)")
  for idx = 1:length(framedSignalsTest)
     [s,f,t] = fsst(framedSignalsTest{idx},Fs,kaiser(128));
     f_indices =  (f > 25) & (f < 200);
     signalsFsstTest{idx}= [real(s(f_indices,:)); imag(s(f_indices,:))];
  end

  signalsFsstTest = cellfun(standardizeFun,signalsFsstTest,'UniformOutput',false);

  %% Train
  net = trainNetwork(signalsFsstTrain,framedLabelsTrain,layers,options);

  kFoldsNets{i} = net;
  figure(9)
  [predTrain, errTrain] = classify(net,signalsFsstTrain,'MiniBatchSize',MiniBatchSize);
  plotconfusion([framedLabelsTrain{:}],[predTrain{:}],'Training')

  C_Train = confusionmat([framedLabelsTrain{:}],[predTrain{:}]);

  % % Test
  figure(10)
  [predTest, errTest] = classify(net,signalsFsstTest,'MiniBatchSize',MiniBatchSize);
  plotconfusion([framedLabelsTest{:}],[predTest{:}],'Testing')

  C_Test = confusionmat([framedLabelsTest{:}],[predTest{:}]);
end
