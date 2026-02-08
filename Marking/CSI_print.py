import csiread

csi_data = csiread.Intel("data/flip50hzR_20s_010.dat")
csi_data.read()

print(csi_data.csi.shape)   # (packets, Nrx, Ntx, subcarriers)
#print(csi_data.csi[0])

print("Total packets:", len(csi_data.csi))



import numpy as np
import matplotlib.pyplot as plt

# CSI 複數矩陣
csi = csi_data.csi   # shape = (packets, subcarriers, Nrx, Ntx)

# 取第1根Rx天線、第1個Tx、所有子載波
amp = np.abs(csi[:, :, 0, 0])


from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff=10, fs=100, order=5):
    """
    cutoff: 截止頻率 (人體動作通常在 10Hz 以下)
    fs: 採樣率 (Intel 5300 的發包頻率，例如 100Hz)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a =  butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# 對所有子載波進行低通濾波
amp_filtered = lowpass_filter(amp, cutoff=1, fs=50)


from sklearn.decomposition import PCA

# 將幅度標準化後進行 PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(amp_filtered)

# 繪製第一主成分 (通常最能反映動作)
plt.plot(pca_result[:, 0])
plt.title("First Principal Component (Motion Feature)")
plt.show()


from scipy.signal import spectrogram
'''
# 使用 PCA 的第一主成分來做頻譜分析
f, t, Sxx = spectrogram(pca_result[:, 0], fs=5, nperseg=64)

plt.pcolormesh(t, f, np.log(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram (Action Signature)')
plt.show()
'''

plt.figure(figsize=(10, 6))
for i in range(amp_filtered.shape[1]):
    plt.plot(amp_filtered[:, i])

plt.legend()
plt.show()