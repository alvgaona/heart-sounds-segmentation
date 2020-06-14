clear
close all

[signals, labels, features] = loadInputData();

data = seperateTrainAndTestData(signals, labels, features, 2000, 0);

frames = framing(data,2000,1000);

[fsstTrain, fsstTest] = extractFeatures(frames, 1000);

[net, predTrain, predTest] = runClassification(frames, fsstTrain, fsstTest);

figure(9)
plotconfusion([frames.framedLabelsTrain{:}],[predTrain{:}],'Training')
figure(10)
plotconfusion([frames.framedLabelsTest{:}],[predTest{:}],'Testing')
