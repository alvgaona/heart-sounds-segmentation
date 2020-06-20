function frames = framing(data, n, stride) 
    [framedSignalsTrain, framedLabelsTrain] = frameSignals(data.signalsTrain, data.labelsTrain, stride, n);
    [framedSignalsVal, framedLabelsVal] = frameSignals(data.signalsVal, data.labelsVal, stride, n);
    [framedSignalsTest, framedLabelsTest] = frameSignals(data.signalsTest, data.labelsTest, stride, n);
    
    [framedFeaturesTrain, ~] = frameSignals(data.featuresTrain, data.labelsTrain, stride, n);
    [framedFeaturesVal, ~] = frameSignals(data.featuresVal, data.labelsVal, stride, n);
    [framedFeaturesTest, ~] = frameSignals(data.featuresTest, data.labelsTest, stride, n);

    frames.framedSignalsTrain = framedSignalsTrain;
    frames.framedLabelsTrain = framedLabelsTrain;
    frames.framedSignalsVal = framedSignalsVal;   
    frames.framedLabelsVal = framedLabelsVal;    
    frames.framedSignalsTest = framedSignalsTest;
    frames.framedLabelsTest = framedLabelsTest;
    frames.framedFeaturesTrain = framedFeaturesTrain;
    frames.framedFeaturesTest = framedFeaturesTest;
    frames.framedFeaturesVal = framedFeaturesVal;
end