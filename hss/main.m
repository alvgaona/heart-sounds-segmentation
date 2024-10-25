clear
close all

signalLen = 2000;

[signals, labels] = loadInputCSV();
lengthSignals = cellfun(@(x) length(x), signals);
signals = signals(lengthSignals >= signalLen);
labels = labels(lengthSignals >= signalLen);

[framedSignals, framedLabels] = frameSignals(signals, labels, 1000, 2000);

groups = kFolds(framedSignals', framedLabels', 10);

[fsstTrain, fsstVal, fsstTest] = extractFeatures(groups{1}, 1000);

trainLabels = groups{1}.train(:,2);
testLabels = groups{1}.test(:,2);

[net, info] = train(groups{1}.train(:,2), fsstTrain, fsstVal, groups{1}.validation(:,2));

[predTrain, trainScores] = classify(net,fsstTrain,'MiniBatchSize',50);
[predTest, testScores] = classify(net,fsstTest,'MiniBatchSize',50);

%% Confusion Matrix
figure
confusionchart([trainLabels{:}],[predTrain{:}],'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Training');
figure
confusionchart([testLabels{:}],[predTest{:}],'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Testing');

%% ROC Curves
labels = [testLabels{:}];
scores = [testScores{:}];
multiClassRocCurve(labels, scores, {'S1','Systole','S2','Diastole'});
