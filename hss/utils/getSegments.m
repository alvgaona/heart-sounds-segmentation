% function segments = getSegments(data, labels)
%
% Get all four states from a PCG.
%
%% INPUTS:
% data: cellarray containing all PCGs
% labels: cellarray containing all PCGs' labels
%
%% OUTPUTS:
% segments: struct containing all states of the PCG dataset
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

function segments = getSegments(data, labels)
  s1 = {};
  sys = {};
  s2 = {};
  dias = {};
  segments = struct();
  state = [];
  for i=1:length(labels)
      % Extract S1 sounds
    if (labels(i) == 1)
      state = [state; data(i)];
      if(i+1 > length(labels))
        s1{end+1} = state;
        state = [];
      else
        if (labels(i+1) ~= 1)
          if (size(s1,1) == 0)
            s1{1} = state;
          else
            s1{end+1} = state;
          end
          state = [];
        end
      end
    end
    % Extract systoles
    if (labels(i) == 2)
      state = [state; data(i)];
      if(i+1 > length(labels))
        sys{end+1} = state;
        state = [];
      else
        if (labels(i+1) ~= 2)
          if (size(sys,1) == 0)
            sys{1} = state;
          else
            sys{end+1} = state;
          end
          state = [];
        end
      end
    end
    % Extract S2 sound
    if (labels(i) == 3)
      state = [state; data(i)];
      if(i+1 > length(labels))
        s2{end+1} = state;
        state = [];
      else
        if (labels(i+1) ~= 3)
          if (size(s2,1) == 0)
            s2{1} = state;
          else
            s2{end+1} = state;
          end
          state = [];
        end
      end
    end
    % Extract diastoles sound
    if (labels(i) == 4)
      state = [state; data(i)];
      if(i+1 > length(labels))
        dias{end+1} = state;
        state = [];
      else
        if (labels(i+1) ~= 4)
          if (size(dias,1) == 0)
            dias{1} = state;
          else
            dias{end+1} = state;
          end
          state = [];
        end
      end
    end
  end
  segments.S1 = s1;
  segments.S2 = s2;
  segments.Sys = sys;
  segments.Dias = dias;
end
