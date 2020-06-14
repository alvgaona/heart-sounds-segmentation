function data = seperateTrainAndTestData(signals, labels, features, signalLen, validationData)   
    if (validationData)
        trainPercentage = 0.7;
        validationPercentage = 0.1;
        testPercentage = 0.2;       
    else
        trainPercentage = 0.7;
        validationPercentage = 0;
        testPercentage = 0.3; 
    end

    rng('shuffle')
    [trainInd, valInd,testInd] = dividerand(length(signals), ...
        trainPercentage, ...
        validationPercentage, ...
        testPercentage);

    signalsTrain = signals(trainInd);
    labelsTrain = labels(trainInd);
    featuresTrain = features(trainInd);
    
    signalsVal = signals(valInd);
    labelsVal = labels(valInd);
    featuresVal = features(valInd);

    signalsTest = signals(testInd);
    labelsTest = labels(testInd);
    featuresTest = features(testInd);

    lengthSignalsTrain = cellfun(@(x) length(x), signalsTrain);
    lengthSignalsTest = cellfun(@(x) length(x), signalsTest);
    lengthSignalsVal = cellfun(@(x) length(x), signalsVal);

    data.signalsTrain = signalsTrain(lengthSignalsTrain >= signalLen);
    data.labelsTrain = labelsTrain(lengthSignalsTrain >= signalLen);
    data.signalsTest = signalsTest(lengthSignalsTest >= signalLen);
    data.labelsTest = labelsTest(lengthSignalsTest >= signalLen);
    data.signalsVal = signalsVal(lengthSignalsVal >= signalLen);
    data.labelsVal = labelsVal(lengthSignalsVal >= signalLen);
    data.featuresTrain = featuresTrain(lengthSignalsTrain >= signalLen);
    data.featuresTest = featuresTest(lengthSignalsTest >= signalLen); 
    data.featuresVal = featuresVal(lengthSignalsVal >= signalLen);
end