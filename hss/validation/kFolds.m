% function groups = kFolds(dataset, labels, k)
%
% Get PCG groups for Cross-validation K-Folds
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

function groups = kFolds(dataset, labels, k)
  if nargin < 3
    k = 5;
  end
  N = floor(length(dataset)/k);
  folds = {};
  for i=0:k-1
    folds{i+1,1} = dataset((i*N)+1:(i+1)*N);
    folds{i+1,2} = labels((i*N)+1:(i+1)*N);
  end
  % Check if there are any observation left without group
  l = mod(length(dataset),k);
  if l ~= 0
    for j=1:l
      fold_idx = mod(j,k);
      folds{fold_idx,1} = [ folds{fold_idx,1} dataset(end-j)];
      folds{fold_idx,2} = [ folds{fold_idx,2} labels(end-j)];
    end
  end
  groups = {};
  logical_array = false(1,k);
  logical_array(2:k) = 1;
  for i=1:k
    test = [ folds{logical_array == 0,1}' folds{logical_array == 0,2}' ];
    train_folds = [ folds(logical_array,1) folds(logical_array,2) ];
    train = [ reshape([ train_folds{:,1} ], [size([ train_folds{:,1} ],1)*size([ train_folds{:,1} ],2),1]) ...
              reshape([ train_folds{:,2} ], [size([ train_folds{:,2} ],1)*size([ train_folds{:,2} ],2),1]) ];
    groups{i,1} = struct();
    groups{i,1}.train = train;
    groups{i,1}.validation = [ test(round(length(test)*0.1):end,1) test(round(length(test)*0.1):end,2) ];
    groups{i,1}.test = [ test(1:end-round(length(test)*0.1)-1,1) test(1:end-round(length(test)*0.1)-1,2) ];
    logical_array = circshift(logical_array,1);
  end
end
