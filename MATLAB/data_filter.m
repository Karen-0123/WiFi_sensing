clc
clear all
warning('off')
csi_trace = read_bf_file('C:\Users\fupei\Desktop\csi\data\flip50hzR_20s_008.dat');
subcarrier=zeros(3,length(csi_trace),30);
for k=1:3
for j=1:30
for i=1:length(csi_trace)
    csi_entry = csi_trace{i};%依次?各??据包，?了平均
    csi = get_scaled_csi(csi_entry);
    csi=csi(1,:,:);%1*3*30
    csi1=squeeze(csi).';% 30*3 complex

    csiabs=db(abs(csi1));%30*3 double%判??矩???size(),中最小值是否?1，不是得?要??一?作?有效值

    csiabs=csiabs(:,k);
       csi1=csi1(:,k);

    subcarrier(k,i,j)=csiabs(j);%j子?波幅度
      if(subcarrier(k,i,j)>=35)
        subcarrier3(k,i,j)=35;
    else if(subcarrier(k,i,j)<=1)
            subcarrier(k,i,j)=1;
        end
    end

end
end
end

y=subcarrier;
subcarrier1=subcarrier(2,:,17); %??某?具体?波
X=subcarrier1';

figure(1)
subplot(2,1,1);
 plot(X); 

%通?db5小波基?行6尺度小波分解 
 [c,l]=wavedec(X,6,'db5'); 
 a1=appcoef(c,l,'db5',1);
 a2=appcoef(c,l,'db5',2); 
 a3=appcoef(c,l,'db5',3); 
 a4=appcoef(c,l,'db5',4); 
 a5=appcoef(c,l,'db5',5); 
 a6=appcoef(c,l,'db5',6); 
 figure(3); 
 subplot(6,1,1);plot(a1);title('尺度1的低?系?'); 
 subplot(6,1,2);plot(a2);title('尺度2的低?系?'); 
 subplot(6,1,3);plot(a3);title('尺度3的低?系?'); 
  subplot(6,1,4);plot(a1);title('尺度4的低?系?');  
   subplot(6,1,5);plot(a1);title('尺度5的低?系?'); 
   subplot(6,1,6);plot(a1);title('尺度6的低?系?'); 
%    
 d1=detcoef(c,l,1);
 d2=detcoef(c,l,2);
 d3=detcoef(c,l,3); 
 d4=detcoef(c,l,4); 
 d5=detcoef(c,l,5); 
 d6=detcoef(c,l,6); 
 figure(4);
 subplot(3,2,1);plot(d1);title('尺度1的高?系?'); 
subplot(3,2,2);plot(d2);title('尺度2的高?系?'); 
subplot(3,2,3);plot(d3);title('尺度3的高?系?'); 
subplot(3,2,4);plot(d4);title('尺度4的高?系?'); 
subplot(3,2,5);plot(d5);title('尺度5的高?系?'); 
subplot(3,2,6);plot(d6);title('尺度6的高?系?'); 

 d6=zeros(1,length(d6));
 d5=zeros(1,length(d5));
 d4=zeros(1,length(d4));
 d3=zeros(1,length(d3));
 d2=zeros(1,length(d2));
 d1=zeros(1,length(d1));
%  
 a1=a1';
a5=a5';
d1=d1';
d2=d2';
d3=d3';
d4=d4';
d5=d5';
d6=d6';

 cA5=appcoef(c,l,'db5',5);
 A5=wrcoef('a',c,l,'db5',5);

 figure(1);
 subplot(2,1,2),
plot(A5);
 hold on 
 title('重构信?');