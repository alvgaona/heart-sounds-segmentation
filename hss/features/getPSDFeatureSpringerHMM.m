%function [psd] = getPSDFeatureSpringerHMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high)
%
% PSD-based feature extraction for heart sound segmentation.
%
%% INPUTS:
% data: this is the audio waveform
% sampling_frequency is self-explanatory
% frequency_limit_low is the lower-bound on the frequency range you want to
% analyse
% frequency_limit_high is the upper-bound on the frequency range
%
%% OUTPUTS:
% psd is the array of maximum PSD values between the max and min limits,
% resampled to the same size as the original data.
%
% This code was developed by David Springer in the paper:
% D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
% Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
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

function psd = getPSDFeatureSpringerHMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high, figures)
  if nargin < 5
    figures = 0;
  end
  % Find the spectrogram of the signal:
  [~,F,T,P] = spectrogram(data,sampling_frequency/40,round(sampling_frequency/80),1:1:round(sampling_frequency/2),sampling_frequency);
  if(figures)
    figure();
    surf(T,F,10*log(P),'edgecolor','none'); axis tight;
    view(0,90);
    xlabel('Time (Seconds)'); ylabel('Hz');
    pause();
  end
  [~, low_limit_position] = min(abs(F - frequency_limit_low));
  [~, high_limit_position] = min(abs(F - frequency_limit_high));
  % Find the mean PSD over the frequency range of interest:
  psd = mean(P(low_limit_position:high_limit_position,:));
  if(figures)
    t4  = (1:length(psd))./sampling_frequency;
    t3  = (1:length(data))./sampling_frequency;
    figure('Name', 'PSD Feature');
    plot(t3,(data - mean(data))./std(data),'c');
    hold on;
    plot(t4, (psd - mean(psd))./std(psd),'k');
    pause();
  end
end
