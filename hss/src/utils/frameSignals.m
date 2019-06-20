% function [framedSignals, framedLabels] = frameSignals(signals, labels, stride, n)
%
% Signal framing for any kind of 1-D signal.
%
%% INPUTS:
% signals: cellarray 1xN
% labels: cellarray 1xN
% stride: step to move the sliding window
% n: length of frame
%
%% OUTPUTS:
% framedSignals: cellarray containing framed signals
% framedLabels: cellarray containing framed labels
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

function [framedSignals, framedLabels] = frameSignals(signals, labels, stride, n)
  framedSignals = [];
  framedLabels = [];
  for k=1:length(signals)
    signal = signals{k};
    signalLabels = labels{k};
    T = length(signal);
    L = floor((T-n-1)/stride);
    for i=0:L
      framedSignal{i+1,1} = signal(:,i*stride+1:i*stride+n);
      framedSignalLabels{i+1,1} = categorical(signalLabels(i*stride+1:i*stride+n))';
    end
    if L <= 0
      framedSignal = { signal(:,1:n) };
      framedSignalLabels = { categorical(signalLabels(1:n))' };
    end
    framedSignals = [ framedSignals; framedSignal ];
    framedLabels = [ framedLabels; framedSignalLabels ];
  end
end
