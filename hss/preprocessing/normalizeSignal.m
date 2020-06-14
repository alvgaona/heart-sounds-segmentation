% function normalizedSignal = normalizeSignal(signal)
%
% This function subtracts the mean and divides by the standard deviation of
% a (1D) signal in order to normalise it for machine learning applications.
%
%% Inputs:
% signal: the original signal
%
%% Outputs:
% normalised_signal: the original signal, minus the mean and divided by
% the standard deviation.
%
% Developed by David Springer for the paper:
% D. Springer et al., ?Logistic Regression-HSMM-based Heart Sound
% Segmentation,? IEEE Trans. Biomed. Eng., In Press, 2015.
%
%% Copyright (C) 2016  David Springer
% dave.springer@gmail.com
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
function normalizedSignal = normalizeSignal(signal)
  meanOfSignal = mean(signal);
  standardDeviation = std(signal);
  normalizedSignal = (signal - meanOfSignal)./standardDeviation;
end
