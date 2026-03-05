hfc = 175; %ºI¤î?²v
lfc = 0;
fs = 1000; %ªö??²v
order = 25;

[b,a] = butter(order, hfc/(fs/2));
figure(3)
freqz(b,a)

dataIn = randn(1000,1);
dataOut = filter(b,a,y);
figure(4)
subplot(2,1,1)
plot(y)
title('original');
axis([0 1000 5 25]);
subplot(2,1,2)
plot(dataOut)
title('afterbutterworth');