clear all
close all
clc

cd('wfdb/mcode/physionet/training/training-a')
[signal, Fs, tm] = rdsamp('filt_a0002');

w = hamming(Fs);
pos = 1;
f = Fs*linspace(0,1,Fs);
x = signal(:,2)-mean(signal(:,2));
figure(1)

for pos=1:(length(x)-Fs)
    h1 = subplot(3,1,1);
    cla(h1)
    plot(tm,x,'b')
    hold on
    plot(tm(pos:pos+Fs-1),w,'r')
    ylim([-1 1])
    subplot(3,1,2)
    xr=x(pos:pos+Fs-1).*w;
    plot(tm,[zeros(1,pos) xr' zeros(1,length(x)-pos-length(xr))])
    ylim([-0.5 0.5]) %muestro hasta los 100Hz
    title(['Señal temporal con ventana. Posición:',num2str(pos)])
    subplot(3,1,3)
    XR=fft(xr,Fs);
    plot(f,abs(XR))
    xlim([0 50]) %muestro hasta los 100Hz
    title('Espectro de 0 a 100Hz')    
    pause(0.0005)
end