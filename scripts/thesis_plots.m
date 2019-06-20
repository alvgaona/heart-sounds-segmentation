clear all
close all
clc

%% Constants
Fs = 1000;

syms x
y1 = piecewise(x<0, 0, x>0, x);
y2 = piecewise(x<0, 0, x>0, 1);

figure(1)
set(gcf,'color','w')
hold on
fplot(y1,[-1,1], 'k-','LineWidth',1)
fplot(y2,[-1,1],'k--','LineWidth',1)
legend('ReLU','Gradient','Location','northwest')

y3 = 1/(1+exp(-x));
y4 = y3*(1-y3);

figure(2)
set(gcf,'color','w')
hold on
fplot(y3,[-5,5], 'k-','LineWidth',1)
fplot(y4,[-5,5], 'k--','LineWidth',1)
legend('Sigmoid','Gradient','Location','northwest')

w = kaiser(128);

figure(1)
set(gcf,'color','w')
plot(w,'k-')
xlabel('Samples')
ylabel('Amplitude')
xlim([0 130])
set(gca,'box','off')

[b,a] = butter(4, 25/(Fs/2), 'high');

figure(2)
set(gcf,'color','w')
freqz(b,a)
set(gca,'box','off')
lines = findall(gcf,'type','line');
lines(1).Color = 'black';
lines(2).Color = 'black';
set(gca,'box','off')
ylim([-120 10])

[b,a] = butter(4, 400/(Fs/2), 'low');

figure(3)
set(gcf,'color','w')
freqz(b,a)
lines = findall(gcf,'type','line');
lines(1).Color = 'black';
lines(2).Color = 'black';
ylim([-120 10])
set(gca,'box','off')

figure(4)
set(gcf,'color','w')
freqz(w)
lines = findall(gcf,'type','line');
lines(1).Color = 'black';
lines(2).Color = 'black';
set(gca,'box','off')
