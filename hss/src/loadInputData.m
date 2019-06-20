function [signals, labels, features] = loadInputData()
    load('hss/src/resources/datasets/david_springer/springer_dataset.mat')
    signals = springer_dataset.audio_data;
    labels = springer_dataset.labels;
    features = springer_dataset.features;

    features = cellfun(@(x) x', features, 'UniformOutput', false);
    signals = cellfun(@(x) x', signals, 'UniformOutput', false);
end