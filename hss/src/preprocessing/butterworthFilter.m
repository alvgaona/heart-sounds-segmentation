% function postLabels = restrictStates(labels)
%
% Butterworth filtering with filtfilt
%
%% INPUTS:
% x: input signal
% Wn: cut-off frequency
% ftype: type of butterworth (low, high, bandpass)
% order: Butterworth filter order
%
%% OUTPUTS:
% y: output signal
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

function y = butterworthFilter(x, Wn, ftype, order)
    if nargin < 4
        order = 1;
    end    
    
   [b,a] = butter(order, Wn, ftype);
    y = filtfilt(b, a, x);
end