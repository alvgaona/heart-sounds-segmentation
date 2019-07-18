% function PCGFeatures = getFeatures(audioData, Fs, downsample, featuresFs)
%
% Description
%
%% INPUTS:
% signals:
% labels:
% stride:
% n:
%
%% OUTPUTS:
% framedSignals:
% framedLabels:
%
%
%% MIT License
%
% Copyright (c) 2019 Alvaro
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
function PCGFeatures = getFeatures(audioData, Fs, downsample, featuresFs)
    if(nargin < 3)
        downsample = false;
        featuresFs = 0;
    end
    % Spike removal from the original paper:
    audioData = spikeRemoval(audioData,Fs);
    % Find the homomorphic envelope
    homomorphicEnvelope = homomorphicEnvelopeWithHilbert(audioData, Fs);
    % Normalise and downsampled (or not) the envelope:
    if downsample
        downsampledHomomorphicEnvelope = resample(homomorphicEnvelope,featuresFs,Fs);
        normalisedHomomorphicEnvelope = normalizeSignal(downsampledHomomorphicEnvelope);
    else
        normalisedHomomorphicEnvelope = normalizeSignal(homomorphicEnvelope);
    end
    %% Hilbert envelope
    hilbertEnv = hilbertEnvelope(audioData, Fs);
    if downsample
        downsampledHilbertEnvelope = resample(hilbertEnv, featuresFs, Fs);
        normalisedHilbertEnvelope = normalizeSignal(downsampledHilbertEnvelope);
    else
        normalisedHilbertEnvelope = normalizeSignal(hilbertEnv);
    end
    %% PSD
    psd = getPSDFeatureSpringerHMM(audioData, Fs, 40,60)';   
    if downsample
       downsampledPsd = resample(psd, length(downsampledHomomorphicEnvelope), length(psd));
       normalisedPsd = normalizeSignal(downsampledPsd);
    else
       normalisedPsd = normalizeSignal(psd);
    end
    normalisedPsd = resample(psd, length(normalisedHomomorphicEnvelope), length(psd));
    %% Wavelet
    wavLevel = 3;
    wavName ='rbio3.9';
    % Audio needs to be longer than 1 second for getDWT to work:
    if(length(audioData)< Fs*1.025)
        audioData = [audioData; zeros(round(0.025*Fs),1)];
    end
    [cD, cA] = getDWT(audioData,wavLevel,wavName);
    wavFeature = abs(cD(wavLevel,:));
    wavFeature = wavFeature(1:length(homomorphicEnvelope));
    if downsample
        downsampledWavelet = resample(wavFeature, featuresFs, Fs);
        normalisedWavelet =  normalizeSignal(downsampledWavelet)';
    else
        normalisedWavelet =  normalizeSignal(wavFeature)';
    end
    PCGFeatures = [normalisedHomomorphicEnvelope, normalisedHilbertEnvelope, normalisedPsd, normalisedWavelet];
end
