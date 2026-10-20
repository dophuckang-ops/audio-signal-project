import streamlit as st 
import librosa 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import io
import os
from tab4_noise_filter import render_noise_filter_tab
from scipy.signal import (
    butter, filtfilt, firwin, freqz, resample_poly, sosfiltfilt,
    sosfreqz, welch,
)
#Cấu hình giao diện ứng dụng
st.set_page_config(page_title="6-STARS | Phân tích và phân biệt âm thanh",layout="wide")
st.title("Phần Mềm Phân Tích Và Xử Lý Tín Hiệu Âm Thanh")
st.subheader("Đề tài: Phân tích và phân biệt âm thanh thông qua đặc trưng tín hiệu miền thời gian và miền tần số")
st.caption("6-STARS")
def load_audio(file):
    file_bytes = file.getvalue()    
    # 1. Ưu tiên đọc trực tiếp trên RAM bằng scipy
    try:
        sr, y = wavfile.read(io.BytesIO(file_bytes))
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        elif y.dtype == np.uint8:
            y = (y.astype(np.float32) - 128.0) / 128.0
        
        if len(y.shape)>1:
            y = np.mean(y, axis=1) # Chuyển Stereo -&gt; Mono
        return y, float(sr)
    except Exception:
        pass

    # 2. Dự phòng: Ghi ra file temp_audio.wav để cấp đuôi .wav cho librosa
    with open("temp_audio.wav", "wb") as f:
        f.write(file_bytes)
    y, sr = librosa.load("temp_audio.wav", sr=None)
    return y, float(sr)
# Ảnh nhóm thực hiện
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "assets/DSC02174.JPG",
        caption="Nhóm thực hiện đề tài phân tích và xử lý tín hiệu âm thanh",
        use_container_width=True
    )
tab1,tab2,tab3,tab4=st.tabs([
    "1.Phân Tích 1 File âm thanh lẻ",
    "2.Trích Xuất Miền Thời Gian",
    "3.Phổ FFT Và Miền Tần Số ",
    "4.Nhiễu Và Lọc Nhiễu"
])
with tab1:
    st.header("Phân tích và xem dạng sóng(waveform) của 1 file âm thanh")
    uploaded_single=st.file_uploader("Chọn 1 File âm thanh(.wav):",type=["wav"],key="single_wav")
    if uploaded_single is not None:
      try:
        y, sr = load_audio(uploaded_single)
        duration = len(y) / sr 
        max_amp = float(np.max(np.abs(y))) 
        rms_energy = np.sqrt(np.mean(y**2))
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        std=np.std(y)
        crest_factor=(max_amp/rms_energy if rms_energy>0 else 0.0)
        st.divider() 
        st.subheader("Nghe Thử Âm Thanh") 
        st.audio(uploaded_single) 
        st.subheader("Thông Số Tín Hiệu")
       #Thông số cơ bản
        c1,c2,c3=st.columns(3)
        c1.metric("Tần số lấy mẫu(SR)",f"{sr:.2f}Hz")
        c2.metric("Thời lượng (Duration)", f"{duration:.2f} s") 
        c3.metric("Tổng số mẫu (N)", f"{len(y):,}")
       #Đặc trưng miền thời gian
        c4,c5,c6,c7,c8=st.columns(5)
        c4.metric("Biên độ cực đại", f"{max_amp:.4f}") 
        c5.metric("Năng lượng RMS", f"{rms_energy:.4f}") 
        c6.metric("Tỷ lệ qua điểm 0 (ZCR)", f"{zcr_mean:.4f}") 
        c7.metric("Độ lệch chuẩn(STD)",f"{std:.4f}")
        c8.metric("Crest Factor", f"{crest_factor:.4f}")
        st.divider()
        st.subheader("Đồ thị dạng sóng(Waveform)")
        times = np.arange(len(y)) / sr 
        fig1, ax1 = plt.subplots(figsize=(11, 3.8)) 
        ax1.plot(times, y, color='#1f77b4', linewidth=0.7, alpha=0.85, label='Waveform') 
        ax1.set_title(f"Waveform: {uploaded_single.name} | SR = {sr} Hz | Duration = {duration:.2f} s", fontweight='bold', pad=10) 
        ax1.set_xlabel("Thời gian (Giây)") 
        ax1.set_ylabel("Biên độ") 
        ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--') 
        ax1.grid(True, linestyle=':', alpha=0.5) 
        plt.tight_layout() 
        st.pyplot(fig1) 
        plt.close(fig1)
      except Exception as e:
         st.error(f" Không thể phân tích file này: {e}")

with tab2:
   #Khối tải file âm thanh
   st.header("Trích Xuất Và Xuất Bảng Đặc Trưng Miền Thời Gian")  
   st.write("Xử lý hàng loạt file âm thanh để tính toán**Peak,RMS,STD,ZCR và xuất file CSV")
   uploaded_time_files=st.file_uploader(
      "Chọn hoặc kéo thả danh sách các file(.wav) cần trích xuất",
      type=["wav"],
      accept_multiple_files=True,
      key="tab2_file_uploader"
   )
   if uploaded_time_files:
      time_results=[]
      #Vòng lặp xử lý từng file âm thanh
      for file in uploaded_time_files:
         try:
            y, sr = load_audio(file)
         except Exception as e:
            st.error(f" Lỗi khi đọc file {file.name}: {e}") 
            continue
        #Tính toán các chỉ số
         duration = float(len(y) / sr) 
         peak_val = float(np.max(np.abs(y))) 
         rms_val = np.sqrt(np.mean(y**2))
         std_val = np.std(y)
         zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y)))
         crest_factor=(peak_val/rms_val if rms_val>0 else 0.0)
        #Tách tên lớp thành chuỗi string chuẩn  
         file_name=file.name
         file_stem = os.path.splitext(file_name)[0]
         class_name = (
         file_stem.split('_')[0].strip().lower()
         if '_' in file_stem
         else "Unknown"
         )
         time_results.append({ 
            "File": file_name, 
            "Class": class_name, 
            "Sample Rate (Hz)": sr, 
            "Duration (s)": round(duration, 2), 
            "Peak": round(peak_val, 5), 
            "RMS": round(rms_val, 5), 
            "STD": round(std_val, 5), 
            "ZCR": round(zcr_val, 5),
            "Crest Factor":round(float(crest_factor),5)})
    #Chuyển đổi dữ liệu thành Dataframe Pandas
      df_time=pd.DataFrame(time_results)
      st.subheader("Bảng Đặc Trưng Miền Thời Gian Chi Tiết")
      st.dataframe(df_time,use_container_width=True)
      csv_time_data=df_time.to_csv(index=False,encoding='utf-8-sig').encode('utf-8-sig')
      st.download_button(
         label="Tải File 'time_features.csv",
         data=csv_time_data,
         file_name="time_features.csv",
         mime="text/csv",
         key="tab2_download_csv_button"
      )
      st.divider()
      #Bảng tổng hợp trung bình theo lớp âm thanh
      st.subheader("Giá Trị Trung Bình Theo Lớp Âm Thanh")
      df_summary=df_time.groupby("Class")[["Peak","RMS","STD","ZCR",]].mean().reset_index()
      class_counts = (
                df_time.groupby("Class")
                .size()
                .reset_index(name="Số lượng file")
            )
      df_summary = df_summary.merge(
                class_counts,
                on="Class",
                how="left"
            )
      df_summary = df_summary[
                [
                    "Class",
                    "Số lượng file",
                    "Peak",
                    "RMS",
                    "STD",
                    "ZCR",
                ]
            ]
      st.dataframe(df_summary.round(8),use_container_width=True)
      st.caption("Bảng tổng hợp hiển thị giá trị trung bình của từng "
                "đặc trưng theo lớp. Số lượng file giúp kiểm tra "
                "mức độ cân bằng dữ liệu.")
      st.divider()
      #Biểu đồ so sánh đặc trưng miền thời gian
      st.subheader("Biểu Đồ So Sánh Đặc Trưng Giữa Các Lớp Âm Thanh")
      col1,col2=st.columns(2)
      with col1:
         fig_rms,ax_rms=plt.subplots(figsize=(5.5,3.5))
         ax_rms.bar(df_summary["Class"],df_summary["RMS"],color='#1f77b4', alpha=0.85)
         ax_rms.set_title("Trung bình RMS Energy theo Lớp", fontweight='bold', fontsize=10)
         ax_rms.set_ylabel("RMS Energy") 
         ax_rms.grid(axis='y', linestyle=':', alpha=0.6) 
         plt.tight_layout() 
         st.pyplot(fig_rms) 
         plt.close(fig_rms) # Đóng figure để tránh giải phóng bộ nhớ đệm
      with col2:
         fig_zcr, ax_zcr = plt.subplots(figsize=(5.5, 3.5)) 
         ax_zcr.bar(df_summary["Class"], df_summary["ZCR"], color='#ff7f0e', alpha=0.85) 
         ax_zcr.set_title("Trung bình ZCR theo Lớp", fontweight='bold', fontsize=10) 
         ax_zcr.set_ylabel("ZCR") 
         ax_zcr.grid(axis='y', linestyle=':', alpha=0.6) 
         plt.tight_layout() 
         st.pyplot(fig_zcr) 
         plt.close(fig_zcr) # Đóng figure để tránh giải phóng bộ nhớ đệm
      st.divider()
      #Biểu đồ phân tán RMS và ZCR
      fig_scatter, ax_scatter=plt.subplots(figsize=(6,4))
      for class_label in df_time["Class"].unique():
         class_data=df_time[df_time["Class"]==class_label]
         ax_scatter.scatter(
            class_data["RMS"],
            class_data["ZCR"],
            label=class_label,
            s=60,
            alpha=0.8
         )
      ax_scatter.set_title("Phân bố RMS và ZCR theo lớp âm thanh",fontweight="bold")
      ax_scatter.set_xlabel("RMS Energy")
      ax_scatter.set_ylabel("Zero-Crossing Rate (ZCR)")
      ax_scatter.grid(True, linestyle=":", alpha=0.6)
      ax_scatter.legend(title="Class")
      plt.tight_layout()
      st.pyplot(fig_scatter)
      plt.close(fig_scatter)
   else:
      st.info("Vui lòng chọn hoặc kéo thả các file'.wav' để trích xuất đặc trưng miền thời gian")
with tab3:
   st.header("Phân tích tín hiệu âm thanh trong miền tần số")
   st.write("Phân tích phổ biên độ, phát hiện các đỉnh phổ và trích xuất đặc trưng miền tần số.")
   uploaded_fft_file=st.file_uploader("Chọn file âm thanh (.wav):",type=["wav"],key="tab3_fft_uploader")
   if uploaded_fft_file is not None:
      try:
         #Đọc tín hiệu
         y,sr=load_audio(uploaded_fft_file)
         y = np.asarray(y, dtype=np.float64)
         if len(y)<2:
            st.error("File âm thanh quá ngắn để phân tích.")
         elif not np.all(np.isfinite(y)):
            st.error("Tín hiệu chứa giá trị không hợp lệ.")
         else:
            #Loại bỏ thành phần DC
            y_centered = y - np.mean(y)
            #Áp dụng cửa sổ Hann
            N = len(y_centered)
            window = np.hanning(N)
            y_windowed = y_centered * window 
            #Tính FFT
            fft_values=np.fft.rfft(y_windowed)
            frequencies=np.fft.rfftfreq(N,d=1/sr)
            #Tính phổ biên độ
            magnitude = (2 * np.abs(fft_values) / np.sum(window))
            magnitude[0] /= 2
            if N % 2 == 0:
               magnitude[-1] /= 2
            #Tìm các đỉnh phổ
            from scipy.signal import find_peaks
            if len(magnitude)>2:
               peak_indices,_=find_peaks(magnitude,prominence=np.max(magnitude)*0.01, distance=max(1,int(20/(sr/N))))
            else:
               peak_indices = np.array([], dtype=int)
            #Loại bỏ đỉnh DC và sắp xếp
            peak_indices = peak_indices[peak_indices > 0]
            peak_indices = peak_indices[np.argsort(magnitude[peak_indices])[::-1]]
            #Tìm tần số trội
            if len(magnitude) > 1:
               dominant_index = (np.argmax(magnitude[1:]) + 1)
               dominant_frequency = frequencies[dominant_index]
               dominant_magnitude = magnitude[dominant_index]
            else:
               dominant_frequency = 0.0
               dominant_magnitude = 0.0
            #Tính các thông số cơ bản
            duration = N / sr
            frequency_resolution = sr / N
            st.divider()
            st.subheader("Thông số tín hiệu")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Tần số lấy mẫu",f"{sr:,}Hz")
            c2.metric("Thời lượng",f"{duration:.3f}s")
            c3.metric("Độ phân giải tần số",f"{frequency_resolution:.3f}Hz")
            c4.metric("Tần số trội",f"{dominant_frequency:.2f}Hz")
            st.divider()
            st.subheader("Phổ biên độ FFT")
            max_frequency = sr / 2
            default_limit = min(5000.0,max_frequency)
            frequency_limit = st.slider("Giới hạn tần số hiển thị (Hz):",min_value=0.0,max_value=float(max_frequency),value=float(default_limit),key="tab3_frequency_limit")
            #Vẽ phổ
            mask = frequencies <= frequency_limit
            fig_fft, ax_fft = plt.subplots(figsize=(11, 4.5))
            ax_fft.plot(frequencies[mask],magnitude[mask],linewidth=0.8,label="Phổ biên độ")
            #Đánh dấu tần số trội nếu nằm trong khoảng
            if dominant_frequency <= frequency_limit:
                    ax_fft.axvline(
                        dominant_frequency,
                        linestyle="--",
                        linewidth=1,
                        label=(
                            f"Tần số trội: "
                            f"{dominant_frequency:.2f} Hz"
                        )
                    )
                    ax_fft.set_title(
                    "Phổ biên độ của tín hiệu âm thanh"
                )
                    ax_fft.set_xlabel("Tần số (Hz)")
                    ax_fft.set_ylabel("Biên độ phổ")
                    ax_fft.grid(
                    True,
                    linestyle=":",
                    alpha=0.5
                )
                    ax_fft.legend()
                    plt.tight_layout()
                    st.pyplot(fig_fft)
                    plt.close(fig_fft)
                    # 12. Bảng đỉnh phổ
                    st.divider()
                    st.subheader("Các đỉnh phổ nổi bật")
                    if len(peak_indices) > 0:
                        peak_table = pd.DataFrame({
                        "Tần số (Hz)": frequencies[peak_indices],
                        "Biên độ phổ": magnitude[peak_indices]
                    })

                    peak_table = peak_table.head(20)

                    st.dataframe(
                        peak_table.round(6),
                        use_container_width=True
                    )

                    csv_peaks = peak_table.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    ).encode("utf-8-sig")

                    st.download_button(
                        "Tải bảng đỉnh phổ (CSV)",
                        data=csv_peaks,
                        file_name="fft_peaks.csv",
                        mime="text/csv",
                        key="tab3_download_peaks"
                    )
            else:
                    st.info(
                        "Không tìm thấy đỉnh phổ nổi bật "
                        "với ngưỡng hiện tại."
                    )
            st.divider()
            st.subheader("Các đặc trưng miền tần số")

             # Dùng bình phương biên độ để tính phân bố năng lượng phổ
            power_spectrum = magnitude ** 2
            total_power = np.sum(power_spectrum)

            if total_power > 0:
                 # Spectral Centroid
                 spectral_centroid = (
                     np.sum(frequencies * magnitude)
                     / np.sum(magnitude)
                     if np.sum(magnitude) > 0
                     else 0.0
                 )

                 # Spectral Bandwidth
                 spectral_bandwidth = np.sqrt(
                     np.sum(
                         power_spectrum
                         * (frequencies - spectral_centroid) ** 2
                     ) / total_power
                 )

                 # Spectral Rolloff tại 85% năng lượng phổ
                 cumulative_power = np.cumsum(power_spectrum)

                 rolloff_index = np.searchsorted(
                     cumulative_power,
                     0.85 * total_power
                 )

                 rolloff_index = min(
                     rolloff_index,
                     len(frequencies) - 1
                 )

                 spectral_rolloff = frequencies[rolloff_index]

                 # Spectral Flatness
                 epsilon = np.finfo(float).eps

                 spectral_flatness = (
                     np.exp(np.mean(np.log(power_spectrum + epsilon)))
                     / (np.mean(power_spectrum) + epsilon)
                 )

            else:
                 spectral_centroid = 0.0
                 spectral_bandwidth = 0.0
                 spectral_rolloff = 0.0
                 spectral_flatness = 0.0
            feature_data = {
                 "Đặc trưng": [
                     "Spectral Centroid",
                     "Spectral Bandwidth",
                     "Spectral Rolloff (85%)",
                     "Spectral Flatness"
                 ],
                 "Giá trị": [
                     spectral_centroid,
                     spectral_bandwidth,
                     spectral_rolloff,
                     spectral_flatness
                 ],
                 "Đơn vị": [
                     "Hz",
                     "Hz",
                     "Hz",
                     "Không thứ nguyên"
                 ],
                 "Ý nghĩa": [
                     "Tâm khối phổ",
                     "Độ phân tán quanh tâm phổ",
                     "Tần số chứa 85% năng lượng phổ tích lũy",
                     "Mức độ phẳng của phổ"
                 ]
             }
            df_frequency_features = pd.DataFrame(feature_data)
            st.dataframe(
                 df_frequency_features.round(6),
                 use_container_width=True
             )
            csv_frequency_features = df_frequency_features.to_csv(
                 index=False,
                 encoding="utf-8-sig"
             ).encode("utf-8-sig")
            st.download_button(
                 "Tải đặc trưng miền tần số (CSV)",
                 data=csv_frequency_features,
                 file_name="frequency_features.csv",
                 mime="text/csv",
                 key="tab3_download_features"
             )
            #Ước lượng tần số cơ bản và phân tích F0
            st.divider()
            st.subheader("Ước lượng tần số cơ bản F0")
            st.write( "Ước lượng F0 để khảo sát cao độ của nguyên âm "
    "và các nốt piano. Một số âm thanh như tiếng vỗ tay "
    "hoặc tiếng gõ có thể không có F0 ổn định.")
            #1. Kiểm tra tín hiệu
            if len(y)==0 or np.max(np.abs(y))<1e-8:
                st.warning("Tín hiệu rỗng hoặc gần như im lặng. "
        "Không thể ước lượng F0.")
            else:
                try:
                    y_f0=np.asarray(y,dtype=np.float32)
                    y_f0=y_f0-np.mean(y_f0)
                    max_abs=np.max(np.abs(y_f0))
                    if max_abs>0:
                        y_f0=y_f0/max_abs
                    frame_lenghth_f0=4096
                    hop_length_f0=256
                    fmin_f0=65
                    fmax_f0=min(1200.0, sr / 2 - 1)
                    if len(y_f0)<frame_lenghth_f0:
                        y_f0=np.pad(y_f0,(0,frame_lenghth_f0-len(y_f0)))
                    if fmax_f0<=fmin_f0:
                        st.warning( "Tần số lấy mẫu quá thấp để ước lượng F0 "
                "trong dải tần đã chọn.")
                    else:
                        f0_values, voiced_flag, voiced_probs = librosa.pyin(y_f0,fmin=fmin_f0,fmax=fmax_f0,sr=int(sr),frame_length=frame_lenghth_f0,hop_length=hop_length_f0)
                        valid_mask=(voiced_flag & np.isfinite(f0_values)& (f0_values>0))
                        if not np.any(valid_mask):
                            st.warning( "Không tìm được F0 đáng tin cậy. "
                    "Âm thanh có thể không có cao độ ổn định, "
                    "quá ngắn hoặc chứa nhiều nhiễu.")
                        else:
                           valid_f0 = f0_values[valid_mask]
                           valid_probs = voiced_probs[valid_mask]
                           confidence_mask = valid_probs >= 0.5
                           if np.any(confidence_mask):
                               reliable_f0 = valid_f0[confidence_mask]
                               estimated_f0 = float(np.median(reliable_f0))
                           else:
                               estimated_f0 = float(np.median(valid_f0))
                           mean_voiced_prob = float(np.mean(valid_probs))
                           voiced_ratio = float(np.mean(valid_mask))
                           mean_voiced_prob = float(np.mean(valid_probs))
                           #Xác định nốt nhạc gần nhất
                           midi_note=int(np.round(librosa.hz_to_midi(estimated_f0)))
                           note_name=librosa.midi_to_note(midi_note)
                           nearest_note_f0=float(librosa.midi_to_hz(midi_note))
                           cents_diviation=float(1200*np.log2(estimated_f0/nearest_note_f0))
                           #Hiển thị kết quả
                           st.markdown("### Kết quả ước lượng")
                           col1,col2,col3 =st.columns(3)
                           col1.metric("Tần số cơ bản F0",f"{estimated_f0:.2f}Hz")
                           col2.metric("Nốt nhạc gần nhất",f"{note_name}")
                           col3.metric("Sai lệch cao độ",f"{cents_diviation:+.1f} cents")
                           st.write(f"Tần số chuẩn của nốt{note_name}:"
                                    f"{nearest_note_f0:.2f} Hz.")
                           col4, col5 = st.columns(2)
                           col4.metric("Tỷ lệ khung hữu thanh",f"{voiced_ratio * 100:.1f}%")
                           col5.metric("Voiced probability trung bình",f"{mean_voiced_prob:.2f}")
                           st.caption(
                    "Voiced probability phản ánh mức độ tin tưởng "
                    "của thuật toán vào việc nhận diện khung hữu thanh; "
                    "không phải xác suất F0 chính xác tuyệt đối."
                )
                           if mean_voiced_prob < 0.5:
                             st.warning(
                        "Kết quả F0 có độ tin cậy hạn chế. "
                        "Nên kiểm tra tín hiệu và phổ FFT trước "
                        "khi đưa ra kết luận."
                    )
                           st.divider()
                           st.markdown("### Đối chiếu nốt piano")
                           piano_reference = {
                    "C4": 261.63,
                    "G4": 392.00,
                    "C5": 523.25
                }
                           reference_note = st.selectbox(
                    "Chọn nốt piano được ghi âm",
                    options=[
                        "Không chọn",
                        "C4",
                        "G4",
                        "C5"
                    ],
                    key="tab3_f0_piano_reference"
                )
                           if reference_note !="Không chọn":
                               target_f0 = piano_reference[reference_note]
                               error_hz = estimated_f0 - target_f0
                               error_percent = (error_hz / target_f0) * 100
                               error_cents = float(1200 * np.log2(estimated_f0 / target_f0))
                               col1, col2, col3 = st.columns(3)
                               col1.metric(  "F0 chuẩn",
                        f"{target_f0:.2f} Hz")
                               col2.metric("Sai lệch tần số",
                        f"{error_hz:+.2f} Hz")
                               col3.metric("Sai lệch cao độ",
                        f"{error_cents:+.1f} cents")
                               st.write(  f"Sai lệch tương đối: "
                        f"{error_percent:+.2f}%.")
                           st.divider()
                           st.markdown("### Phân tích họa âm (Harmonics)")
                           st.write( "Tần số lý thuyết của họa âm bậc k: "
                    "fₖ = k × F0.")
                           harmonic_data=[]
                           num_harmonics = 10
                           tolerance=max(10,frequency_resolution*3)
                           for k in range(1,num_harmonics + 1):
                               theoretical_freq = k * estimated_f0
                               if theoretical_freq>=sr/2:
                                   break
                               mask=( (frequencies >= theoretical_freq - tolerance)
                        & (frequencies <= theoretical_freq + tolerance))
                               if not np.any(mask):
                                   continue
                               local_indices=np.where(mask)[0]
                               peak_idx=local_indices[np.argmax(magnitude[local_indices])]
                               actual_freq = float(frequencies[peak_idx])
                               peak_magnitude = float(magnitude[peak_idx])
                               frequency_error=(actual_freq - theoretical_freq)
                               harmonic_data.append({
                        "Bậc họa âm": k,
                        "Tần số lý thuyết (Hz)": round(
                            theoretical_freq, 2
                        ),
                        "Tần số đỉnh phổ (Hz)": round(
                            actual_freq, 2
                        ),
                        "Sai lệch (Hz)": round(
                            frequency_error, 2
                        ),
                        "Biên độ": round(
                            peak_magnitude, 6
                        )
                    })
                           #Hiển thị kết quả và xuất bảng hoạ âm
                           if harmonic_data:
                               df_harmonics=pd.DataFrame(harmonic_data)
                               st.dataframe(df_harmonics,
                        use_container_width=True,
                        hide_index=True)
                               csv_harmonics = df_harmonics.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                               st.download_button(
                        label="Tải bảng họa âm (CSV)",
                        data=csv_harmonics,
                        file_name="harmonic_analysis.csv",
                        mime="text/csv",
                        key="tab3_f0_harmonics_csv"
                    )
                               st.caption(
                        "Các đỉnh phổ được tìm trong vùng lân cận "
                        "bội số của F0. Kết quả là phép đối chiếu "
                        "gần đúng, không đảm bảo mọi đỉnh tìm thấy "
                        "đều là họa âm thực sự."
                    )
                           else:
                               st.info(
                        "Không tìm thấy đỉnh phổ phù hợp "
                        "với các họa âm trong dải phân tích."
                    )
                except Exception as e:
                    st.error(
            f"Không thể ước lượng F0 hoặc phân tích họa âm: {e}"
        )
                    # 13. Nghe lại âm thanh
            st.divider()
            st.subheader("Nghe lại file âm thanh")
            st.audio(uploaded_fft_file)
      except Exception as e:
            st.error(
                f"Không thể phân tích file âm thanh: {e}"
            )
   else:
        st.info(
            "Vui lòng tải lên một file WAV để bắt đầu phân tích."
        )
with tab4:
    render_noise_filter_tab()
    



      


   

         
