"""6-STARS: phân tích âm thanh, so sánh đặc trưng và lọc nhiễu.
Bản sửa tách hàm tính toán dùng chung để bảng và đồ thị dùng cùng định nghĩa.
"""
import io
import os
import re
from pathlib import Path
import librosa
import matplotlib
matplotlib.use("Agg")  # Vẽ hình cho Streamlit, không mở cửa sổ đồ họa riêng.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import find_peaks
from tab4_noise_filter import render_noise_filter_tab

@st.cache_data(show_spinner=False, max_entries=64)
def load_audio_bytes(data: bytes):
    """Read WAV bytes without temporary files; preserve recording amplitude."""
    y, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    if y.shape[0] < 32:
        raise ValueError("File âm thanh phải có ít nhất 32 mẫu.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Tín hiệu chứa NaN hoặc giá trị vô hạn.")
    if int(sr) <= 0:
        raise ValueError("Tần số lấy mẫu không hợp lệ.")
    return np.mean(y, axis=1, dtype=np.float64), int(sr)


def parse_audio_label(filename):
    """Preserve vowel subclasses and separate a piano note from its class."""
    stem = os.path.splitext(os.path.basename(filename))[0].strip().casefold()
    piano_match = re.fullmatch(
        r"piano[\s_-]+([a-g](?:#|b)?-?\d+)(?:[\s_-]+\d+)?", stem
    )
    if piano_match:
        note = piano_match.group(1)
        return "piano", note[0].upper() + note[1:]
    label = re.sub(r"[\s_-]+\d+$", "", stem).strip()
    label = re.sub(r"[\s-]+", "_", label).strip("_")
    return label or "unknown", ""


def _validated_mono(y, sr):
    signal = np.asarray(y, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Cần tín hiệu mono không rỗng.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Tín hiệu chứa NaN hoặc giá trị vô hạn.")
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError("Tần số lấy mẫu phải lớn hơn 0.")
    return signal


def time_features(y, sr):
    """Full-file amplitude statistics; ZCR is crossings per second."""
    signal = _validated_mono(y, sr)
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    nonnegative = signal >= 0
    crossings = int(np.count_nonzero(nonnegative[1:] != nonnegative[:-1]))
    return {
        "Peak": peak,
        "Peak-to-peak": float(np.ptp(signal)),
        "RMS": rms,
        "Mean (DC)": float(np.mean(signal)),
        "STD": float(np.std(signal)),
        "ZCR (1/s)": float(crossings * float(sr) / signal.size),
        "Crest Factor": peak / rms if rms > 0 else float("nan"),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def estimate_f0(
    data: bytes,
    fmin=65.0,
    fmax=1200.0,
    probability_threshold=0.5,
    min_reliable_ratio=0.1,
):
    """Estimate a file's F0 only when enough voiced, energetic frames pass the criteria.

    mean_probability averages pYIN probabilities across ALL frames.
    reliable_ratio and voiced_ratio use ALL analysed frames as denominator.
    RMS gating applies to the mean-removed, peak-normalized pitch copy only.
    """
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 < fmin < fmax):
        raise ValueError("Cần 0 < F0 tối thiểu < F0 tối đa.")
    if not 0 <= probability_threshold <= 1 or not 0 <= min_reliable_ratio <= 1:
        raise ValueError("Ngưỡng xác suất và tỷ lệ khung phải nằm trong [0, 1].")

    y, sr = load_audio_bytes(data)
    analysis_sr, frame_length, hop_length = 16000, 2048, 256
    pitch = y - np.mean(y)
    if sr != analysis_sr:
        pitch = librosa.resample(pitch, orig_sr=sr, target_sr=analysis_sr)
    total_frames = max(0, 1 + (pitch.size - frame_length) // hop_length)
    times = (np.arange(total_frames, dtype=float) * hop_length + frame_length / 2) / analysis_sr
    result = {
        "f0_hz": float("nan"),
        "reliable_ratio": 0.0,
        "voiced_ratio": 0.0,
        "mean_probability": 0.0,
        "reliable_frames": 0,
        "total_frames": int(total_frames),
        "status": "Chưa có F0 đủ tin cậy.",
        "times": times,
        "contour": np.full(total_frames, np.nan),
        "frame_length": frame_length,
        "hop_length": hop_length,
        "analysis_sr": analysis_sr,
    }
    if total_frames == 0:
        result["status"] = "Âm thanh quá ngắn: cần ít nhất 128 ms cho một khung F0; không chèn thêm mẫu."
        return result
    peak = float(np.max(np.abs(pitch)))
    rms = float(np.sqrt(np.mean(pitch ** 2)))
    if peak <= 1e-8 or rms <= 1e-8:
        result["status"] = "Tín hiệu im lặng, gần im lặng hoặc chỉ có thành phần DC; F0 không xác định."
        return result

    effective_fmax = min(float(fmax), analysis_sr / 2 - 1, sr / 2 - 1)
    if effective_fmax <= fmin:
        result["status"] = "Tần số lấy mẫu không đủ cho dải F0 đã chọn."
        return result
    pitch = pitch / peak
    f0, voiced_flag, probabilities = librosa.pyin(
        pitch,
        fmin=float(fmin),
        fmax=effective_fmax,
        sr=analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
        fill_na=np.nan,
    )
    frame_rms = librosa.feature.rms(
        y=pitch,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    if len(f0) != total_frames or len(frame_rms) != total_frames:
        raise RuntimeError("Số khung F0 và RMS không khớp; cần kiểm tra phiên bản thư viện.")
    voiced = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    reliable = (
        voiced
        & np.isfinite(probabilities)
        & (probabilities >= probability_threshold)
        & (frame_rms >= 0.01)
    )
    reliable_count = int(np.count_nonzero(reliable))
    finite_probabilities = np.where(np.isfinite(probabilities), probabilities, 0.0)
    result.update({
        "reliable_ratio": float(np.mean(reliable)),
        "voiced_ratio": float(np.mean(voiced)),
        "mean_probability": float(np.mean(finite_probabilities)) if finite_probabilities.size else 0.0,
        "reliable_frames": reliable_count,
    })
    if reliable_count < 3 or result["reliable_ratio"] < min_reliable_ratio:
        result["status"] = (
            f"F0 không xác định: {reliable_count}/{total_frames} khung đạt ngưỡng "
            f"({result['reliable_ratio']:.1%}); cần ít nhất 3 khung và {min_reliable_ratio:.0%}."
        )
        return result

    result["f0_hz"] = float(np.median(f0[reliable]))
    result["contour"] = np.where(reliable, f0, np.nan)
    result["status"] = "Đạt tiêu chí tin cậy; F0 là trung vị của các khung được chấp nhận."
    return result


def fft_analysis(y, sr):
    """One-sided Hann amplitude spectrum, with consistent amplitude moments.

    Rolloff uses 85% of squared spectral amplitude; flatness also uses power.
    A constant or silent signal has no defined dominant frequency.
    """
    signal = _validated_mono(y, sr)
    if signal.size < 3:
        raise ValueError("Cần ít nhất 3 mẫu để áp dụng cửa sổ Hann.")
    n = signal.size
    frequencies = np.fft.rfftfreq(n, d=1 / float(sr))
    centered = signal - np.mean(signal)
    if np.ptp(signal) == 0:
        centered = np.zeros_like(signal)
    window = np.hanning(n)
    magnitude = 2 * np.abs(np.fft.rfft(centered * window)) / np.sum(window)
    magnitude[0] /= 2
    if n % 2 == 0:
        magnitude[-1] /= 2
    power = magnitude ** 2
    total_power = float(np.sum(power))
    magnitude_sum = float(np.sum(magnitude))
    result = {
        "frequencies": frequencies,
        "magnitude": magnitude,
        "peak_indices": np.array([], dtype=int),
        "dominant_frequency": float("nan"),
        "dominant_magnitude": 0.0,
        "frequency_resolution": float(sr) / n,
        "spectral_centroid": float("nan"),
        "spectral_bandwidth": float("nan"),
        "spectral_rolloff": float("nan"),
        "spectral_flatness": float("nan"),
    }
    if total_power <= 0 or magnitude_sum <= 0:
        return result
    if np.any(magnitude[1:] > 0):
        dominant_index = int(np.argmax(magnitude[1:])) + 1
        result["dominant_frequency"] = float(frequencies[dominant_index])
        result["dominant_magnitude"] = float(magnitude[dominant_index])
    peak_indices, _ = find_peaks(
        magnitude,
        prominence=float(np.max(magnitude)) * 0.01,
        distance=max(1, int(20 / result["frequency_resolution"])),
    )
    peak_indices = peak_indices[peak_indices > 0]
    result["peak_indices"] = peak_indices[np.argsort(magnitude[peak_indices])[::-1]]
    centroid = float(np.sum(frequencies * magnitude) / magnitude_sum)
    bandwidth = float(np.sqrt(np.sum(magnitude * (frequencies - centroid) ** 2) / magnitude_sum))
    rolloff_index = min(int(np.searchsorted(np.cumsum(power), 0.85 * total_power)), len(frequencies) - 1)
    # Scaling before the logarithm keeps flatness independent of recording gain.
    normalized_power = power / np.max(power)
    flatness = float(
        np.exp(np.mean(np.log(np.maximum(normalized_power, np.finfo(float).tiny))))
        / np.mean(normalized_power)
    )
    result.update({
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": float(frequencies[rolloff_index]),
        "spectral_flatness": flatness,
    })
    return result


def show_figure(fig, filename, key):
    """Hiện hình và lưu PNG từ cùng một figure để hình báo cáo khớp giao diện."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    st.pyplot(fig)
    st.download_button("Tải hình PNG", buffer.getvalue(), filename,
                       "image/png", key=key)
    plt.close(fig)


def download_csv(frame, filename, key, label="Tải bảng CSV"):
    # Chỉ làm tròn khi hiển thị, giữ độ chính xác đầy đủ trong CSV.
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       filename, "text/csv", key=key)


@st.cache_data(show_spinner=False, max_entries=128)
def file_feature_row(data, filename, f0_settings):
    y, sr = load_audio_bytes(data)
    class_name, note = parse_audio_label(filename)
    row = {"File": filename, "Class": class_name, "Note": note,
           "Sample Rate (Hz)": sr, "Duration (s)": len(y) / sr,
           **time_features(y, sr)}
    spectrum = fft_analysis(y, sr)
    for label, field in [
        ("Dominant frequency (Hz)", "dominant_frequency"),
        ("Spectral centroid (Hz)", "spectral_centroid"),
        ("Spectral bandwidth (Hz)", "spectral_bandwidth"),
        ("Spectral rolloff 85% (Hz)", "spectral_rolloff"),
        ("Spectral flatness", "spectral_flatness"),
    ]:
        row[label] = spectrum[field]
    pitch = estimate_f0(data, **f0_settings)
    row.update({
        "F0 (Hz)": pitch["f0_hz"], "F0 status": pitch["status"],
        "Reliable F0 frames": pitch["reliable_frames"],
        "Total F0 frames": pitch["total_frames"],
        "Reliable F0 ratio": pitch["reliable_ratio"],
        "Voiced ratio": pitch["voiced_ratio"],
        "Mean voiced probability": pitch["mean_probability"],
        "F0 analysis SR (Hz)": pitch["analysis_sr"],
        "F0 frame length": pitch["frame_length"],
        "F0 hop length": pitch["hop_length"],
        "F0 min (Hz)": f0_settings["fmin"],
        "F0 max requested (Hz)": f0_settings["fmax"],
        "F0 probability threshold": f0_settings["probability_threshold"],
        "F0 minimum reliable ratio": f0_settings["min_reliable_ratio"],
    })
    return row


def render_single_tab():
    st.header("Phân tích một file âm thanh")
    uploaded = st.file_uploader("Chọn file WAV", type=["wav"], key="single_wav")
    if uploaded is None:
        st.info("Chọn file WAV để nghe và xem dạng sóng.")
        return
    try:
        y, sr = load_audio_bytes(uploaded.getvalue())
        values = time_features(y, sr)
        st.audio(uploaded.getvalue(), format="audio/wav")
        a, b, c = st.columns(3)
        a.metric("Tần số lấy mẫu", f"{sr:,} Hz")
        b.metric("Thời lượng", f"{len(y) / sr:.3f} s")
        c.metric("Số mẫu", f"{len(y):,}")
        st.dataframe(pd.DataFrame([values]).round(6), width="stretch", hide_index=True)
        st.caption(
            "RMS là biên độ hiệu dụng. Các chỉ số tính trên toàn bản ghi mono, "
            "không chuẩn hóa từng file theo Peak. ZCR dùng số lần đổi dấu mỗi giây; "
            "mẫu bằng 0 được xem là không âm."
        )
        if values["Peak"] >= 0.999:
            st.warning("Tín hiệu mono gần/chạm full scale; nên kiểm tra clipping.")
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(np.arange(len(y)) / sr, y, linewidth=0.7)
        ax.set(title=f"Waveform: {uploaded.name}", xlabel="Thời gian (s)",
               ylabel="Biên độ số")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        show_figure(fig, "waveform.png", "tab1_waveform_png")
        download_csv(pd.DataFrame([{"File": uploaded.name,
                                   "Sample Rate (Hz)": sr, **values}]),
                     "single_features.csv", "tab1_csv")
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Không thể phân tích file: {exc}")


def class_summary(frame):
    """SD ở đây đo biến thiên GIỮA bản ghi; khác STD của mẫu trong một file."""
    features = ["Peak", "RMS", "STD", "ZCR (1/s)", "Crest Factor", "F0 (Hz)"]
    grouped = frame.groupby("Class", sort=True)
    summary = grouped[features].agg(["mean", "std"])
    summary.columns = [f"{name} - {'TB' if stat == 'mean' else 'SD giữa file'}"
                       for name, stat in summary.columns]
    summary.insert(0, "Số file", grouped.size())
    summary.insert(1, "Số file có F0", grouped["F0 (Hz)"].count())
    return summary.reset_index()


def scatter_figure(frame, y_column, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Màu của một lớp giữ nguyên kể cả khi lớp đó không có F0 hợp lệ.
    all_classes = sorted(frame["Class"].unique())
    colors = plt.get_cmap("tab10")
    for i, label in enumerate(all_classes):
        points = frame[(frame["Class"] == label)
                       & np.isfinite(frame["RMS"]) & np.isfinite(frame[y_column])]
        if points.empty:
            continue
        ax.scatter(points["RMS"], points[y_column], label=label,
                   color=colors(i % 10), s=55, alpha=0.8)
        if y_column == "F0 (Hz)":
            for _, point in points.iterrows():
                if point["Note"]:
                    ax.annotate(point["Note"], (point["RMS"], point[y_column]),
                                xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel="Biên độ RMS", ylabel=y_column)
    ax.grid(alpha=0.3)
    if ax.collections:
        ax.legend(title="Lớp âm")
    fig.tight_layout()
    return fig


def render_batch_tab(f0_settings):
    st.header("Trích xuất đặc trưng và so sánh các lớp âm thanh")
    st.write("Mỗi hàng và mỗi điểm trên đồ thị tương ứng một file WAV.")
    uploads = st.file_uploader("Chọn danh sách file WAV", type=["wav"],
                               accept_multiple_files=True, key="tab2_file_uploader")
    if not uploads:
        st.info("Chọn các file WAV để tạo bảng và đồ thị so sánh.")
        return
    rows, errors = [], []
    with st.spinner("Đang trích xuất đặc trưng; lần đầu tính F0 có thể mất thời gian..."):
        for upload in uploads:
            try:
                rows.append(file_feature_row(upload.getvalue(), upload.name, f0_settings))
            except (ValueError, RuntimeError, OSError) as exc:
                errors.append({"File": upload.name, "Lỗi": str(exc)})
    if errors:
        st.warning(f"Có {len(errors)} file không xử lý được; xem nguyên nhân bên dưới.")
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
    if not rows:
        st.info("Chưa có file hợp lệ để tổng hợp. Hãy kiểm tra các file đã chọn.")
        return
    frame = pd.DataFrame(rows)
    st.subheader("Bảng đặc trưng hợp nhất")
    st.dataframe(frame.round(6), width="stretch", hide_index=True)
    download_csv(frame, "features.csv", "tab2_features_csv", "Tải features.csv")
    st.caption(
        "RMS/Peak/STD tính trên toàn bản ghi mono ở SR gốc. ZCR là số lần đổi dấu "
        "mỗi giây (không còn là tỷ lệ trên mẫu). Để so sánh công bằng, cần thống nhất "
        "điều kiện thu, dải tần và quy tắc chọn đoạn/khoảng im lặng. "
        "Chỉ bản dùng ước lượng F0 được đổi SR; CSV lưu thông số phân tích."
    )
    summary = class_summary(frame)
    st.subheader("Trung bình và biến thiên giữa các lần thu")
    st.dataframe(summary.round(6), width="stretch", hide_index=True)
    download_csv(summary, "class_summary.csv", "tab2_summary_csv")
    st.caption(
        "SD giữa file dùng ddof=1 và để trống khi chỉ có một giá trị hợp lệ. "
        "STD trong bảng từng file mô tả độ phân tán mẫu của chính tín hiệu đó. "
        "Trung bình F0 chỉ dùng các file có F0 đủ tin cậy."
    )
    for column, feature in zip(st.columns(2), ["RMS", "ZCR (1/s)"]):
        with column:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar(summary["Class"], summary[f"{feature} - TB"],
                   yerr=summary[f"{feature} - SD giữa file"].fillna(0), capsize=4)
            ax.set(title=f"{feature}: trung bình ± SD giữa file", ylabel=feature)
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            slug = "rms" if feature == "RMS" else "zcr"
            show_figure(fig, f"class_{slug}.png", f"tab2_bar_{slug}")

    st.subheader("Phân bố các đặc trưng theo từng file")
    show_figure(scatter_figure(frame, "ZCR (1/s)", "RMS–ZCR theo lớp âm thanh"),
                "rms_zcr.png", "tab2_scatter_zcr")
    left, right = st.columns(2)
    with left:
        show_figure(scatter_figure(frame, "Peak", "RMS–Peak theo lớp âm thanh"),
                    "rms_peak.png", "tab2_scatter_peak")
        st.caption("Peak/RMS là Crest Factor. Các điểm mô tả dữ liệu đã thu, không phải độ chính xác phân loại.")
    with right:
        valid = np.isfinite(frame["F0 (Hz)"])
        st.caption(f"RMS–F0 dùng {int(valid.sum())}/{len(frame)} file đủ tin cậy; "
                   f"loại {int((~valid).sum())} file khỏi đồ thị này.")
        if valid.any():
            show_figure(scatter_figure(frame, "F0 (Hz)", "RMS–F0 theo lớp âm thanh"),
                        "rms_f0.png", "tab2_scatter_f0")
        else:
            st.info("Không có F0 đủ tin cậy để vẽ RMS–F0; các đặc trưng khác vẫn được giữ.")
        if (~valid).any():
            st.dataframe(frame.loc[~valid, ["File", "F0 status"]],
                         width="stretch", hide_index=True)


def render_frequency_tab(f0_settings):
    """Tab 3: FFT, đặc trưng phổ và F0 có kiểm tra độ tin cậy."""
    st.header("Phân tích tín hiệu âm thanh trong miền tần số")
    st.write(
        "Quan sát phổ biên độ, phát hiện đỉnh phổ và ước lượng cao độ. "
        "Tần số có biên độ lớn nhất trên phổ không nhất thiết là F0."
    )
    uploaded = st.file_uploader(
        "Chọn file âm thanh (.wav):", type=["wav"], key="tab3_fft_uploader"
    )
    if uploaded is None:
        st.info("Vui lòng chọn file WAV để bắt đầu phân tích.")
        return

    data = uploaded.getvalue()
    try:
        y, sr = load_audio_bytes(data)
        spectrum = fft_analysis(y, sr)
    except Exception as exc:
        st.error(f"Không thể đọc hoặc phân tích phổ của file: {exc}")
        return

    frequencies = np.asarray(spectrum["frequencies"], dtype=float)
    magnitude = np.asarray(spectrum["magnitude"], dtype=float)
    peak_indices = np.asarray(spectrum["peak_indices"], dtype=int)
    dominant_frequency = float(spectrum["dominant_frequency"])
    frequency_resolution = float(spectrum["frequency_resolution"])

    st.subheader("Thông số tín hiệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tần số lấy mẫu", f"{sr:,.0f} Hz")
    c2.metric("Thời lượng", f"{len(y) / sr:.3f} s")
    c3.metric("Khoảng cách các bin FFT", f"{frequency_resolution:.3f} Hz")
    c4.metric(
        "Tần số trội",
        f"{dominant_frequency:.2f} Hz"
        if np.isfinite(dominant_frequency) else "Không xác định",
    )
    st.caption(
        "Δf = SR/N là khoảng cách giữa các bin FFT. Khả năng tách hai tần số "
        "gần nhau còn phụ thuộc cửa sổ và độ dài đoạn thu."
    )

    st.divider()
    st.subheader("Phổ biên độ FFT")
    max_frequency = float(sr / 2)
    frequency_limit = st.slider(
        "Giới hạn tần số hiển thị (Hz)",
        min_value=0.0, max_value=max_frequency,
        value=min(5000.0, max_frequency),
        key=f"tab3_frequency_limit_{int(sr)}",
    )
    visible = frequencies <= frequency_limit
    fig_fft, ax_fft = plt.subplots(figsize=(11, 4.5))
    ax_fft.plot(
        frequencies[visible], magnitude[visible], linewidth=0.8,
        label="Phổ biên độ một phía",
    )
    # Chỉ dấu đánh dấu phụ thuộc phạm vi; hình và các bảng luôn được hiển thị.
    if np.isfinite(dominant_frequency) and dominant_frequency <= frequency_limit:
        ax_fft.axvline(
            dominant_frequency, linestyle="--", linewidth=1,
            label=f"Tần số trội: {dominant_frequency:.2f} Hz",
        )
    ax_fft.set(
        title="FFT sau khi bỏ DC và áp dụng cửa sổ Hann",
        xlabel="Tần số (Hz)", ylabel="Biên độ số",
        xlim=(0, max(frequency_limit, frequency_resolution)),
    )
    ax_fft.grid(True, linestyle=":", alpha=0.5)
    ax_fft.legend()
    fig_fft.tight_layout()
    show_figure(fig_fft, "fft_spectrum.png", "tab3_fft_png")
    st.caption(
        "Thanh giới hạn chỉ thay đổi vùng nhìn của hình. Đỉnh phổ và đặc trưng "
        "bên dưới vẫn được tính trên toàn dải từ 0 đến SR/2."
    )

    st.subheader("Các đỉnh phổ nổi bật")
    # Khởi tạo cả khi không có đỉnh; file im lặng không làm lỗi bảng/CSV.
    peak_indices = peak_indices[np.argsort(magnitude[peak_indices])[::-1]]
    peak_table = pd.DataFrame({
        "Tần số (Hz)": frequencies[peak_indices],
        "Biên độ phổ": magnitude[peak_indices],
    })
    if peak_table.empty:
        st.info("Không tìm thấy đỉnh phổ nổi bật với ngưỡng phát hiện hiện tại.")
    else:
        st.dataframe(peak_table.head(20).round(6), width="stretch", hide_index=True)
        st.caption(
            f"Hiển thị tối đa 20 đỉnh mạnh nhất trong {len(peak_table)} đỉnh. "
            "CSV chứa toàn bộ các đỉnh được phát hiện."
        )
    st.download_button(
        "Tải bảng đỉnh phổ (CSV)",
        data=peak_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="fft_peaks.csv", mime="text/csv", key="tab3_download_peaks",
    )

    st.divider()
    st.subheader("Các đặc trưng miền tần số")
    features = pd.DataFrame({
        "Đặc trưng": [
            "Spectral Centroid", "Spectral Bandwidth",
            "Spectral Rolloff (85%)", "Spectral Flatness",
        ],
        "Giá trị": [
            spectrum["spectral_centroid"], spectrum["spectral_bandwidth"],
            spectrum["spectral_rolloff"], spectrum["spectral_flatness"],
        ],
        "Đơn vị": ["Hz", "Hz", "Hz", "Không thứ nguyên"],
        "Ý nghĩa": [
            "Tâm phổ, dùng trọng số biên độ",
            "Độ phân tán quanh tâm phổ, dùng trọng số biên độ",
            "Tần số đạt 85% tổng bình phương biên độ phổ tích lũy",
            "Trung bình nhân/trung bình cộng của bình phương biên độ phổ",
        ],
    })
    st.dataframe(features.round(6), width="stretch", hide_index=True)
    st.caption(
        "Các giá trị không xác định được để trống/NaN, không gán 0. "
        "Centroid và bandwidth dùng cùng trọng số biên độ để nhất quán. "
        "Chỉ bảng hiển thị được làm tròn; CSV giữ độ chính xác tính toán."
    )
    st.download_button(
        "Tải đặc trưng miền tần số (CSV)",
        data=features.to_csv(index=False).encode("utf-8-sig"),
        file_name="frequency_features.csv", mime="text/csv",
        key="tab3_download_features",
    )

    st.divider()
    st.subheader("Ước lượng tần số cơ bản F0")
    st.write(
        "pYIN ước lượng F0 theo từng khung thời gian. Chỉ tổng hợp các khung "
        "hữu thanh đủ tin cậy và chỉ nhận kết quả khi tỷ lệ khung đó đạt ngưỡng "
        "đã chọn. Tiếng vỗ tay hoặc âm nhiễu có thể không có F0 ổn định."
    )
    try:
        f0_result = estimate_f0(data, **f0_settings)
        estimated_f0 = float(f0_result["f0_hz"])
        reliable_ratio = float(f0_result["reliable_ratio"])
        voiced_ratio = float(f0_result["voiced_ratio"])
        mean_probability = float(f0_result["mean_probability"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Tỷ lệ khung hữu thanh", f"{voiced_ratio:.1%}")
        c2.metric("Tỷ lệ khung đủ tin cậy", f"{reliable_ratio:.1%}")
        c3.metric(
            "Voiced probability trung bình",
            f"{mean_probability:.3f}" if np.isfinite(mean_probability)
            else "Không xác định",
        )
        st.caption(
            f"Khung đủ tin cậy: {f0_result['reliable_frames']}/"
            f"{f0_result['total_frames']}. Ngưỡng probability: "
            f"{f0_settings['probability_threshold']:.2f}; tỷ lệ tối thiểu: "
            f"{f0_settings['min_reliable_ratio']:.1%}. "
            "Voiced probability không phải xác suất F0 đúng tuyệt đối."
        )
        st.caption(
            f"F0 được phân tích tại SR = {f0_result['analysis_sr']:,.0f} Hz; "
            f"cửa sổ {f0_result['frame_length']} mẫu; "
            f"bước khung {f0_result['hop_length']} mẫu. "
            "Đây là cấu hình phân tích F0, tách biệt với FFT của bản ghi gốc."
        )

        if not np.isfinite(estimated_f0) or estimated_f0 <= 0:
            st.warning(
                "Không có F0 đủ tin cậy cho đoạn thu này. "
                f"Trạng thái: {f0_result['status']}. "
                "Không suy ra nốt nhạc hoặc họa âm từ kết quả bị từ chối."
            )
        else:
            # Chỉ chạy phần suy diễn cao độ khi F0 đã vượt các tiêu chí chấp nhận.
            midi_note = int(np.round(librosa.hz_to_midi(estimated_f0)))
            note_name = librosa.midi_to_note(midi_note)
            nearest_note_f0 = float(librosa.midi_to_hz(midi_note))
            cents_deviation = float(1200 * np.log2(estimated_f0 / nearest_note_f0))
            c1, c2, c3 = st.columns(3)
            c1.metric("F0 tổng hợp", f"{estimated_f0:.2f} Hz")
            c2.metric("Nốt nhạc gần nhất", note_name)
            c3.metric("Sai lệch so với nốt gần nhất", f"{cents_deviation:+.1f} cents")
            st.caption(
                f"Tần số chuẩn {note_name}: {nearest_note_f0:.2f} Hz, "
                "theo hệ bình quân 12 cung với A4 = 440 Hz. F0 tổng hợp là "
                "trung vị các khung được chấp nhận; cần xem diễn biến F0 "
                "nếu đoạn thu thay đổi cao độ."
            )

            times = np.asarray(f0_result["times"], dtype=float)
            contour = np.asarray(f0_result["contour"], dtype=float)
            if times.shape == contour.shape and np.any(np.isfinite(contour)):
                fig_f0, ax_f0 = plt.subplots(figsize=(11, 3))
                ax_f0.plot(times, contour, ".-", markersize=2, linewidth=0.7)
                ax_f0.axhline(
                    estimated_f0, linestyle="--", color="tab:orange",
                    label=f"F0 tổng hợp: {estimated_f0:.2f} Hz",
                )
                ax_f0.set(
                    title="F0 theo thời gian ở các khung đủ tin cậy",
                    xlabel="Thời gian (s)", ylabel="F0 (Hz)",
                )
                ax_f0.grid(alpha=0.3)
                ax_f0.legend()
                fig_f0.tight_layout()
                show_figure(fig_f0, "f0_contour.png", "tab3_f0_png")

            st.subheader("Đối chiếu nốt piano")
            reference_note = st.selectbox(
                "Chọn nốt piano được ghi âm",
                ["Không chọn", "C4", "G4", "C5"],
                key="tab3_f0_piano_reference",
            )
            if reference_note != "Không chọn":
                # Tính chuẩn từ thang âm, không dùng hằng số Hz đã làm tròn.
                target_f0 = float(librosa.note_to_hz(reference_note))
                error_hz = estimated_f0 - target_f0
                error_percent = 100 * error_hz / target_f0
                error_cents = float(1200 * np.log2(estimated_f0 / target_f0))
                c1, c2, c3 = st.columns(3)
                c1.metric("F0 chuẩn", f"{target_f0:.2f} Hz")
                c2.metric("Sai lệch tần số", f"{error_hz:+.2f} Hz")
                c3.metric("Sai lệch cao độ", f"{error_cents:+.1f} cents")
                st.write(f"Sai lệch tương đối: {error_percent:+.2f}%.")

            st.subheader("Đối chiếu các đỉnh ứng viên họa âm")
            tolerance = max(10.0, 3 * frequency_resolution)
            st.write(
                "Với mỗi bậc k, đối chiếu tần số lý thuyết k × F0 với đỉnh "
                f"phổ đã phát hiện trong khoảng ±{tolerance:.2f} Hz."
            )
            harmonic_rows = []
            used_indices = set()
            for harmonic_order in range(1, 11):
                theoretical_frequency = harmonic_order * estimated_f0
                if theoretical_frequency >= sr / 2:
                    break
                candidates = [
                    int(idx) for idx in peak_indices
                    if abs(frequencies[idx] - theoretical_frequency) <= tolerance
                    and int(idx) not in used_indices
                ]
                # Chỉ nhận cực đại thực do bộ dò tìm thấy; không chọn bin bất kỳ.
                if candidates:
                    chosen = min(
                        candidates,
                        key=lambda idx: abs(frequencies[idx] - theoretical_frequency),
                    )
                    used_indices.add(chosen)
                    actual_frequency = float(frequencies[chosen])
                    actual_magnitude = float(magnitude[chosen])
                    frequency_error = actual_frequency - theoretical_frequency
                    match_status = "Có đỉnh ứng viên"
                else:
                    actual_frequency = actual_magnitude = frequency_error = np.nan
                    match_status = "Không có đỉnh phù hợp"
                harmonic_rows.append({
                    "Bậc k": harmonic_order,
                    "Tần số lý thuyết k×F0 (Hz)": theoretical_frequency,
                    "Tần số đỉnh ứng viên (Hz)": actual_frequency,
                    "Sai lệch (Hz)": frequency_error,
                    "Biên độ phổ": actual_magnitude,
                    "Trạng thái": match_status,
                })
            harmonics = pd.DataFrame(harmonic_rows)
            if not harmonics.empty:
                st.dataframe(harmonics.round(6), width="stretch", hide_index=True)
                st.download_button(
                    "Tải bảng ứng viên họa âm (CSV)",
                    data=harmonics.to_csv(index=False).encode("utf-8-sig"),
                    file_name="harmonic_candidates.csv", mime="text/csv",
                    key="tab3_f0_harmonics_csv",
                )
                st.caption(
                    "Ô trống là chưa tìm thấy đỉnh phù hợp, không phải biên độ 0. "
                    "Mỗi đỉnh chỉ gán cho tối đa một bậc. Đây là các ứng viên "
                    "gần bội số F0, chưa chứng minh chắc chắn là họa âm. "
                    "Đối chiếu với F0 trung vị có thể kém phù hợp nếu cao độ "
                    "thay đổi trong bản ghi."
                )
            else:
                st.info("Không có bậc họa âm nào nằm dưới tần số Nyquist.")
    except Exception as exc:
        # Lỗi F0 không che mất các kết quả FFT và khả năng nghe lại bản thu.
        st.error(f"Không thể hoàn tất phân tích F0: {exc}")

    st.divider()
    st.subheader("Nghe lại file âm thanh")
    st.audio(data, format="audio/wav")


def main():
    st.set_page_config(page_title="6-STARS | Phân tích và phân biệt âm thanh", layout="wide")
    st.title("Phần Mềm Phân Tích Và Xử Lý Tín Hiệu Âm Thanh")
    st.subheader("Đề tài: Phân tích và phân biệt âm thanh thông qua đặc trưng tín hiệu miền thời gian và miền tần số")
    st.caption("6-STARS")
    image_path = Path(__file__).resolve().parent /"DSC02174.JPG"
    if image_path.is_file():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.image(str(image_path), caption="Nhóm thực hiện đề tài", width="stretch")
    with st.sidebar:
        st.header("Cấu hình F0 dùng chung")
        fmin = st.number_input("F0 tối thiểu (Hz)", 50.0, 800.0, 65.0, key="f0_min")
        fmax = st.number_input("F0 tối đa (Hz)", 100.0, 4000.0, 1200.0, key="f0_max")
        probability = st.slider("Ngưỡng voiced probability", 0.1, 0.95, 0.5, 0.05, key="f0_probability")
        ratio = st.slider("Tỷ lệ khung tin cậy tối thiểu", 0.05, 0.8, 0.1, 0.05, key="f0_ratio")
        st.caption("Cùng cấu hình được áp dụng ở tab 2 và tab 3. Mặc định cần ít nhất 3 khung và 10% tổng khung đạt ngưỡng; RMS mỗi khung phải đạt 1% Peak của bản dùng phân tích F0.")
        st.caption("Đây là tiêu chí sàng lọc do nhóm lựa chọn, không bảo đảm F0 đúng tuyệt đối. File ngắn hoặc ít khung đạt tiêu chí sẽ được ghi là không xác định.")
    settings = {"fmin": fmin, "fmax": fmax,
                "probability_threshold": probability, "min_reliable_ratio": ratio}
    valid_settings = fmin < fmax
    if not valid_settings:
        st.error("F0 tối thiểu phải nhỏ hơn F0 tối đa. Hãy chỉnh cấu hình bên trái.")
    tab1, tab2, tab3, tab4 = st.tabs([
        "1.Phân Tích 1 File âm thanh lẻ",
        "2.Trích Xuất Và So Sánh Đặc Trưng",
        "3.Phổ FFT Và Miền Tần Số",
        "4.Nhiễu Và Lọc Nhiễu",
    ])
    with tab1:
        render_single_tab()
    with tab2:
        if valid_settings:
            render_batch_tab(settings)
    with tab3:
        if valid_settings:
            render_frequency_tab(settings)
    with tab4:
        render_noise_filter_tab()


if __name__ == "__main__":
    main()
