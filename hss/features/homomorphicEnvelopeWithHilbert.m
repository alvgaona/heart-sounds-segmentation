% function homomorphic_envelope = homomorphicEnvelopeWithHilbert(input_signal, sampling_frequency,lpf_frequency,figures)
%
% This function finds the homomorphic envelope of a signal, using the method
% described in the following publications:
%
% S. E. Schmidt et al., ?Segmentation of heart sound recordings by a 
% duration-dependent hidden Markov model.,? Physiol. Meas., vol. 31, no. 4,
% pp. 513?29, Apr. 2010.
% 
% C. Gupta et al., ?Neural network classification of homomorphic segmented
% heart sounds,? Appl. Soft Comput., vol. 7, no. 1, pp. 286?297, Jan. 2007.
%
% D. Gill et al., ?Detection and identification of heart sounds using 
% homomorphic envelogram and self-organizing probabilistic model,? in 
% Computers in Cardiology, 2005, pp. 957?960.
% (However, these researchers found the homomorphic envelope of shannon
% energy.)
%
% In I. Rezek and S. Roberts, ?Envelope Extraction via Complex Homomorphic
% Filtering. Technical Report TR-98-9,? London, 1998, the researchers state
% that the singularity at 0 when using the natural logarithm (resulting in
% values of -inf) can be fixed by using a complex valued signal. They
% motivate the use of the Hilbert transform to find the analytic signal,
% which is a converstion of a real-valued signal to a complex-valued
% signal, which is unaffected by the singularity. 
%
% A zero-phase low-pass Butterworth filter is used to extract the envelope.
%% Inputs:
% input_signal: the original signal (1D) signal
% samplingFrequency: the signal's sampling frequency (Hz)
% lpf_frequency: the frequency cut-off of the low-pass filter to be used in
% the envelope extraciton (Default = 8 Hz as in Schmidt's publication).
% figures: (optional) boolean variable dictating the display of a figure of
% both the original signal and the extracted envelope:
%
%% Outputs:
% homomorphic_envelope: The homomorphic envelope of the original
% signal (not normalised).
%
% This code was developed by David Springer for comparison purposes in the
% paper:
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

function homomorphic_envelope = homomorphicEnvelopeWithHilbert(input_signal, sampling_frequency,lpf_frequency,figures)

if nargin < 4
    figures = 0;
end
if nargin < 3
    figures = 0;
    lpf_frequency = 8;
end

%8Hz, 1st order, Butterworth LPF
[B_low,A_low] = butter(1,2*lpf_frequency/sampling_frequency,'low');
homomorphic_envelope = exp(filtfilt(B_low,A_low,log(abs(hilbert(input_signal)))));

% Remove spurious spikes in first sample:
homomorphic_envelope(1) = [homomorphic_envelope(2)];

if(figures)
    figure('Name', 'Homomorphic Envelope');
    plot(input_signal);
    hold on;
    plot(homomorphic_envelope,'r');
    legend('Original Signal','Homomorphic Envelope')
end
