clear
close all

[signals, labels, features] = loadInputData();

data = seperateTrainAndTestData(signals, labels, features, 2000, 1);

frames = framing(data,2000,1000);

[fsstTrain, fsstVal, fsstTest] = extractFeatures(frames, 1000);

net = train(frames, fsstTrain, fsstVal, frames.framedLabelsVal);

[predTrain, trainScores] = classify(net,fsstTrain,'MiniBatchSize',50);
[predTest, testScores] = classify(net,fsstTest,'MiniBatchSize',50);

%% Confusion Matrix
figure
confusionchart([frames.framedLabelsTrain{:}],[predTrain{:}],'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Training');
figure
confusionchart([frames.framedLabelsTest{:}],[predTest{:}],'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Testing');

%% ROC Curves
labels = [frames.framedLabelsTest{:}];
scores = [testScores{:}];
multiClassRocCurve(labels, scores, {'S1','Systole','S2','Diastole'});



