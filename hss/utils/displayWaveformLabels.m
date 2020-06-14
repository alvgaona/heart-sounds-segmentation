% function displayWaveformLabels(y,c)
%
% Plots 1-D signal with labels passed.
%
%% INPUTS:
% y: time-series data
% c: labels
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

function displayWaveformLabels(y, c)
    x = 1:length(y);
    custom_colormap = [
                        .8500 .3250 .0980; ... % red
                        1     .5    0; ... % orange
                        0     .4470 .7410; ... % blue
                        .25   .25   .25; ... % black
                       ];
    
    xx=[x; x];           %// create a 2D matrix based on "X" column
    yy=[y; y];           %// same for Y
    zz=zeros(size(xx)); %// everything in the Z=0 plane
    cc =[c; c] ;         %// matrix for "CData"
    
    surf(xx,yy,zz,cc,'EdgeColor','interp','FaceColor','none','LineStyle','-','LineWidth',2);
    colormap(custom_colormap) ;     %// assign the colormap
    shading flat                    %// so each line segment has a plain color
    view(2) %// view(0,90)          %// set view in X-Y plane
    colorbar('Location','eastoutside', ...
             'Ticks',[1.3,2.1,2.9,3.6], ...
             'TickLabels',{'S1','Systole','S2','Diastole'}, ...
             'FontSize', 14, ...
             'Direction', 'reverse')
end