function frames = framing(data, n, stride) 
    [framedSignalsTrain, framedLabelsTrain] = frameSignals(data.signalsTrain, data.labelsTrain, stride, n);
    [framedSignalsTest, framedLabelsTest] = frameSignals(data.signalsTest, data.labelsTest, stride, n);

    [framedFeaturesTrain, ~] = frameSignals(data.featuresTrain, data.labelsTrain, stride, n);
    [framedFeaturesTest, ~] = frameSignals(data.featuresTest, data.labelsTest, stride, n);

    frames.framedSignalsTrain = framedSignalsTrain;
    frames.framedLabelsTrain = framedLabelsTrain;
    frames.framedSignalsTest = framedSignalsTest;
    frames.framedLabelsTest = framedLabelsTest;
    frames.framedFeaturesTrain = framedFeaturesTrain;
    frames.framedFeaturesTest = framedFeaturesTest;
end