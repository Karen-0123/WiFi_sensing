%clc
%clear all;
%數據預處理
csi_trace = read_bf_file('C:\Users\fupei\Desktop\csi\data\flip50hzR_20s_008.dat');
antenna=zeros(3,length(csi_trace));
 j=2;%子載波序號選擇
 for k=1:3
   for i=1:length(csi_trace);
     csi_entry=csi_trace{i};
    csi=get_scaled_csi(csi_entry);
    csi1=squeeze(csi(1,:,:)).';% 30*3 complex

    csiabs=db(abs(csi1));

       csiabs=csiabs(:,k);
       csi1=csi1(:,k);
    subcarrier(i)=csiabs(j);%10子載波幅度

  if(subcarrier(i)>=25)
        subcarrier(i)=25;
    else if(subcarrier(i)<=1) %若采集的數据產生了無窮值或者异常值可用該語句限幅
            subcarrier(i)=1;
        end
  end

   end
 antenna(k,:)=subcarrier;
     figure(k)

     %hampel
x=subcarrier;
[y,i,xmedian,xsigma] = hampel(x,10,4);% 每四個點值取平均，超出三倍的絕對中位差被認為是异常值
n = 1:length(csi_trace);
   figure(k)
plot(n,x)
hold on
 plot(n,xmedian-3*xsigma,n,xmedian+3*xsigma)
plot(find(i),x(i),'sr')
hold off
legend('Original signal','Lower limit','Upper limit','Outliers')
end