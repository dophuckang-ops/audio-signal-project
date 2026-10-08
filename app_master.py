import streamlit as st 
import librosa 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import io
import os
#Cấu hình giao diện ứng dụng
st.set_page_config(page_title="Audio Signal Processing Suite",layout="wide")
st.title("Phần Mềm Phân Tích Và Xử Lý Tín Hiệu Âm Thanh")
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
tab1,tab2,tab3=st.tabs([
    "1.Phân Tích 1 File âm thanh lẻ",
    "2.Trích xuất Miền Thời Gian",
    "3.Phổ FFT Và Miền Tần Số "
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
        c4,c5,c6,c7=st.columns(4)
        c4.metric("Biên độ cực đại", f"{max_amp:.4f}") 
        c5.metric("Năng lượng RMS", f"{rms_energy:.4f}") 
        c6.metric("Tỷ lệ qua điểm 0 (ZCR)", f"{zcr_mean:.4f}") 
        c7.metric("Độ lệch chuẩn(STD)",f"{std:.4f}")
        st.divider()
        st.subheader("Đồ thị dạng sóng(Waveform)")
        times = np.linspace(0, duration, len(y)) 
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
            st.error(f"❌ Lỗi khi đọc file {file.name}: {e}") 
            continue
        #Tính toán các chỉ số
         duration = float(len(y) / sr) 
         peak_val = float(np.max(np.abs(y))) 
         rms_val = np.sqrt(np.mean(y**2))
         std_val = np.std(y)
         zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        #Tách tên lớp thành chuỗi string chuẩn
         file_name=file.name
         class_name=file_name.split('_')[0] if'_'in file_name else "Unknown"
         time_results.append({ 
            "File": file_name, 
            "Class": class_name, 
            "Sample Rate (Hz)": sr, 
            "Duration (s)": round(duration, 2), 
            "Peak": round(peak_val, 5), 
            "RMS": round(rms_val, 5), 
            "STD": round(std_val, 5), 
            "ZCR": round(zcr_val, 5) })
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
      df_summary=df_time.groupby("Class")[["Peak","RMS","STD","ZCR"]].mean().reset_index()
      st.dataframe(df_summary,use_container_width=True)
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
      fig_scatter, ax_scatter=plt.subplots(figsize=(8,5))
      for class_name in df_time(["Class"]).unique():
         class_data=df_time[df_time["Class"]==class_name]
         ax_scatter.scatter(
            class_data["RMS"],
            class_data["ZCR"],
            label=class_name,
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
         
      


   

         
