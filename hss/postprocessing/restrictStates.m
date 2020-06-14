% function postLabels = restrictStates(labels)
%
% Postprocessing algorithm to restrict states transitions
%
%% INPUTS:
% labels: 1xN
%
%% OUTPUTS:
% postLabels: 1xN
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

function postLabels = restrictStates(labels)
  labels = grp2idx(labels);
  postLabels = [];
  postLabels(1) = labels(1);
  for i=2:length(labels)
    if (labels(i) == mod(postLabels(i-1),4)+1)
      postLabels(i) = labels(i);
    else
      postLabels(i) = postLabels(i-1);
    end
  end
  postLabels = categorical(postLabels);
end
   