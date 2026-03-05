%clc
%clear all;
%數據讀取
csi_trace = read_bf_file('C:\Users\fupei\Desktop\csi\data\flip50hzR_20s_001.dat');
for j=1:3
  for i=1:length(csi_trace)%這里是取的數据包的個數
    csi_entry = csi_trace{i};
    csi = get_scaled_csi(csi_entry); %提取csi矩陣    
    csi =csi(1,:,:);
    csi1=abs(squeeze(csi).');          

    %天線選擇

    ant_csi(:,i)=csi1(:,j);             

  end
  figure(j);
  plot(ant_csi.');
  hold on
end