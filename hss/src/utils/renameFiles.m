% function [framedSignals, framedLabels] = frameSignals(signals, labels, stride, n)
%
% Renames files based on path and pattern.
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

function renameFiles(path, pattern)
    if ~isfolder(path)
      errorMessage = sprintf('Error: The following path does not exist:\n%s', path);
      uiwait(warndlg(errorMessage));
      return;
    end

    filePattern = fullfile(path, pattern);
    files = dir(filePattern);

    prefix = 's';
    ext = '';

    for i=1:length(files)
      baseFileName = files(i).name;
      splitStr = strsplit(baseFileName,'.');
      ext = splitStr{2};
      splitStr = splitStr{1};
      splitStr = strsplit(splitStr,'s');
      num = str2num(splitStr{2});

      newFileName = strcat('s',num2str(sprintf( '%04d', num)),'.',ext);

      fullFileName = fullfile(path, baseFileName);
      newFullFileName = fullfile(path, newFileName);
      fprintf('Now renaming %s to\n %s\n', fullFileName, newFullFileName)

      movefile(fullFileName, newFullFileName)

    end
end
