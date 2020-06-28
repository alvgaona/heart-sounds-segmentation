% classdef Metrics < matlab.mixin.SetGet
%
% Properties:
%
% description
% confusionMatrix
% TPR
% TNR
% PPV
% NPV
% FNR
% FPR
% FDR
% FOR
% ACC
% F1
% MCC
% BM
% MK
%
% Methods:
%
% function obj = ClassMetrics(confusionMatrix,description)
% function truePositiveRate(obj)
% function positivePredictiveValue(obj)
% function positivePredictiveValue(obj)
% function negativePredictiveValue(obj)
% function falseNegativeRate(obj)
% function falsePositiveRate(obj)
% function falseDiscoveryRate(obj)
% function falseOmissionRate(obj)
% function accuracy(obj)
% function f1Score(obj)
% function matthewCorrelationCoefficient(obj)
% function bookMakerInformedness(obj)
% function markedness(obj)
% function calculateMetrics(obj)
%
% Class to calculate classification metrics based on https://en.wikipedia.org/wiki/Confusion_matrix
%
% MIT License
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
classdef Metrics < matlab.mixin.SetGet
  properties
    description
    confusionMatrix
    TPR
    TNR
    PPV
    NPV
    FNR
    FPR
    FDR
    FOR
    ACC
    F1
    MCC
    BM
    MK
  end
    
  properties (Access = private)
    TP
    TN
    FP
    FN
  end

  methods
    function obj = Metrics(confusionMatrix,description)
       obj.confusionMatrix = confusionMatrix;
       obj.description = description;
    end

    calculateMetrics(obj)
  end

  methods (Access = private)
    truePositiveRate(obj)

    trueNegativeRate(obj)

    positivePredictiveValue(obj)

    negativePredictiveValue(obj)

    falseNegativeRate(obj)

    falsePositiveRate(obj)

    falseDiscoveryRate(obj)

    falseOmissionRate(obj)

    accuracy(obj)

    f1Score(obj)

    matthewCorrelationCoefficient(obj)

    bookMakerInformedness(obj)

    markedness(obj)
  end
end
