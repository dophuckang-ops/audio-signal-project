import io
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import (
    butter, filtfilt, firwin, freqz, resample_poly, sosfiltfilt,
    sosfreqz, welch,
)


# 1. ĐỌC VÀ KIỂM TRA DỮ LIỆU
def read_wav(data):
    """Đọc WAV trong RAM, đổi stereo thành mono, giữ nguyên sample rate."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    if sr < 1000 or len(audio) < 32:
        raise ValueError("Cần WAV có SR ≥ 1000 Hz và ít nhất 32 mẫu.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("File chứa mẫu NaN hoặc vô cực.")
    return audio.mean(axis=1), int(sr)


def rms(x):
    """Biên độ hiệu dụng; RMS² mới là trung bình bình phương."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def reference_ratio_db(reference, estimate):
    """Tỷ số tín hiệu/sai số; sai số gồm cả nhiễu lẫn méo so với tham chiếu."""
    signal_power = np.mean(reference ** 2)
    error_power = np.mean((estimate - reference) ** 2)
    if signal_power <= 0:
        return np.nan
    if error_power == 0:
        return np.inf
    return float(10 * np.log10(signal_power / error_power))


def mix_at_snr(reference, noise, snr_db):
    """Nhân nhiễu với hệ số để đạt SNR yêu cầu, không chuẩn hóa riêng tín hiệu."""
    reference = np.asarray(reference, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if reference.shape != noise.shape or reference.ndim != 1:
        raise ValueError("Tín hiệu và nhiễu phải là hai mảng mono cùng chiều dài.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(noise)):
        raise ValueError("Dữ liệu trộn phải hữu hạn.")
    noise = noise - np.mean(noise)  # Bỏ DC của riêng nhiễu trước khi trộn.
    if rms(reference) == 0 or rms(noise) == 0:
        raise ValueError("Tham chiếu hoặc nhiễu không có năng lượng để trộn.")
    scale = rms(reference) / (rms(noise) * 10 ** (snr_db / 20))
    scaled_noise = scale * noise
    return reference + scaled_noise, scaled_noise


# 2. THIẾT KẾ VÀ ÁP DỤNG BỘ LỌC
def design_filter(sr, kind, family, cutoff, order=4, numtaps=201):
    """Thiết kế một lượt lọc. Cả FIR lẫn IIR sẽ được áp dụng hai chiều."""
    edges = np.atleast_1d(cutoff).astype(float)
    required = 2 if kind == "bandpass" else 1
    if kind not in ("lowpass", "highpass", "bandpass"):
        raise ValueError("Loại lọc không hợp lệ.")
    if (len(edges) != required or not np.all(np.isfinite(edges))
            or np.any(edges <= 0) or np.any(edges >= sr / 2)
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("Cần 0 < tần số cắt < SR/2; thông dải cần f_low < f_high.")
    if family == "IIR":
        if not 1 <= order <= 10:
            raise ValueError("Bậc thiết kế IIR phải từ 1 đến 10.")
        # SOS = các khâu bậc hai, ổn định số hơn đa thức bậc cao.
        coefficients = butter(order, cutoff, btype=kind, fs=sr, output="sos")
        zeros = min(
            np.sum(coefficients[:, 2] == 0),
            np.sum(coefficients[:, 5] == 0),
        )
        padlen = int(3 * (2 * len(coefficients) + 1 - zeros))
    elif family == "FIR":
        if numtaps < 3 or numtaps % 2 == 0:
            raise ValueError("Số hệ số FIR phải là số lẻ ≥ 3.")
        # FIR cửa sổ Hamming có hệ số đối xứng và pha tuyến tính một lượt.
        coefficients = firwin(
            numtaps, cutoff, pass_zero=kind, fs=sr, window="hamming",
        )
        padlen = 3 * len(coefficients)
    else:
        raise ValueError("Chọn FIR hoặc IIR.")
    return coefficients, padlen


def apply_filter(x, family, coefficients, padlen):
    """Lọc tiến rồi lùi: pha bằng 0, dùng cho file đã ghi, không phải realtime."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("Tín hiệu lọc phải là mono và chứa các mẫu hữu hạn.")
    if len(x) <= padlen + 1:
        raise ValueError(
            f"File quá ngắn: cần ít nhất {padlen + 2} mẫu; "
            "hãy giảm bậc/số hệ số hoặc dùng file dài hơn."
        )
    if family == "IIR":
        result = sosfiltfilt(coefficients, x, padlen=padlen)
    else:
        result = filtfilt(coefficients, [1.0], x, padlen=padlen)
    if not np.all(np.isfinite(result)):
        raise ValueError("Kết quả lọc không hữu hạn; hãy kiểm tra tham số lọc.")
    return result


def response_curve(sr, family, coefficients):
    """Trả đáp ứng biên độ hiệu dụng hai chiều, không phải đáp ứng một lượt."""
    if family == "IIR":
        frequencies, h = sosfreqz(coefficients, worN=4096, fs=sr)
    else:
        frequencies, h = freqz(coefficients, worN=4096, fs=sr)
    # |H_hai_chieu| = |H|², nên 20 log10(|H|²) = 40 log10(|H|).
    db = 40 * np.log10(np.maximum(np.abs(h), 1e-6))
    return frequencies, db


def wav_bytes(x, sr, gain):
    """Cùng gain cho mọi bản phát/tải; không thay đổi dữ liệu dùng đo chỉ số."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(x) * gain, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# 3. GIAO DIỆN TAB 4
def render_noise_filter_tab():
    st.header("Nhiễu và lọc nhiễu bằng bộ lọc số")
    st.write(
        "Chọn dải tần cần giữ, thiết kế FIR/IIR, rồi so sánh tín hiệu trước/sau. "
        "Bộ lọc chỉ suy giảm theo tần số; không tự nhận biết đâu là âm mong muốn."
    )
    mode = st.radio(
        "Thí nghiệm",
        ["Lọc bản ghi có sẵn", "Trộn nhiễu có tham chiếu"],
        key="t4_mode",
    )
    experiment = mode == "Trộn nhiễu có tham chiếu"
    source_file = st.file_uploader(
        "File tham chiếu WAV" if experiment else "File WAV cần lọc",
        type=["wav"], key="t4_source",
    )
    if source_file is None:
        st.info("Chọn một file WAV để bắt đầu.")
        return

    try:
        source, sr = read_wav(source_file.getvalue())
        reference = None
        noise_name = ""
        snr_target = np.nan
        seed = None
        if experiment:
            st.caption(
                "Tham chiếu được coi là tín hiệu cần giữ. Nếu bản thu vốn có nhiễu, "
                "phép đo chỉ đánh giá khả năng phục hồi bản thu đó."
            )
            noise_file = st.file_uploader(
                "File chỉ chứa nhiễu, ví dụ tiếng máy sấy; bỏ trống để tạo nhiễu trắng",
                type=["wav"], key="t4_noise",
            )
            snr_target = st.slider("SNR khi trộn (dB)", -10, 30, 10, key="t4_snr")
            if noise_file is not None:
                noise, noise_sr = read_wav(noise_file.getvalue())
                if noise_sr != sr:
                    divisor = math.gcd(noise_sr, sr)
                    noise = resample_poly(noise, sr // divisor, noise_sr // divisor)
                n = min(len(source), len(noise))
                reference, noise = source[:n], noise[:n]
                noise_name = noise_file.name
                st.caption(f"Hai file dùng {n / sr:.3f} s đầu chung, tại SR = {sr} Hz.")
            else:
                seed = int(st.number_input(
                    "Seed tạo nhiễu (giữ cố định khi so bộ lọc)",
                    min_value=0, max_value=1000000, value=42, step=1, key="t4_seed",
                ))
                reference = source.copy()
                noise = np.random.default_rng(seed).standard_normal(len(source))
                noise_name = "White Gaussian noise"
            before, _ = mix_at_snr(reference, noise, snr_target)
        else:
            before = source.copy()

        st.caption(f"SR: {sr} Hz | Thời lượng dùng: {len(before) / sr:.3f} s")
        if np.max(np.abs(source)) >= 0.999:
            st.warning("Bản thu gần/chạm full scale; hãy kiểm tra clipping trước khi kết luận.")

        family = st.selectbox("Họ bộ lọc", ["IIR", "FIR"], key="t4_family")
        labels = {
            "Lowpass - thông thấp": "lowpass",
            "Highpass - thông cao": "highpass",
            "Bandpass - thông dải": "bandpass",
        }
        kind_label = st.selectbox("Loại bộ lọc", list(labels), key="t4_kind")
        kind = labels[kind_label]
        order, numtaps = 4, 201
        if family == "IIR":
            order = st.slider("Bậc thiết kế Butterworth N", 1, 10, 4, key="t4_order")
        else:
            numtaps = st.slider(
                "Số hệ số FIR (lẻ); bậc một lượt = số hệ số − 1",
                31, 1001, 201, step=2, key="t4_taps",
            )

        # Đổi key theo SR để giá trị đã chọn ở file cũ không vượt Nyquist file mới.
        upper = float(sr / 2 - 1)
        if kind == "bandpass":
            low = st.number_input(
                "Tần số cắt dưới (Hz)", 1.0, upper, min(100.0, upper / 4),
                key=f"t4_low_{sr}",
            )
            high = st.number_input(
                "Tần số cắt trên (Hz)", 1.0, upper, min(3000.0, upper * 0.8),
                key=f"t4_high_{sr}",
            )
            cutoff = [low, high]
        else:
            default = 3000.0 if kind == "lowpass" else 100.0
            cutoff = st.number_input(
                "Tần số cắt danh định (Hz)", 1.0, upper, min(default, upper * 0.8),
                key=f"t4_cutoff_{kind}_{sr}",
            )
        st.caption(
            "Chọn tần số cắt dựa trên phổ của âm cần giữ và nhiễu. "
            "Giá trị mặc định chỉ là điểm bắt đầu để thử nghiệm."
        )
        coefficients, padlen = design_filter(sr, kind, family, cutoff, order, numtaps)
        after = apply_filter(before, family, coefficients, padlen)
        if family == "IIR":
            single_order = order * (2 if kind == "bandpass" else 1)
        else:
            single_order = numtaps - 1
        st.caption(
            f"Bậc một lượt: {single_order}; bậc hiệu dụng hai chiều: {2 * single_order}. "
            "Xử lý ngoại tuyến, pha bằng 0; hai đầu bản ghi vẫn có thể bị ảnh hưởng."
        )

        # 4. ĐO TRÊN DỮ LIỆU GỐC, TRƯỚC KHI ĐỔI GAIN ĐỂ NGHE
        signals = {"Trước lọc": before, "Sau lọc": after}
        if reference is not None:
            signals = {"Tham chiếu": reference, **signals}
        common_peak = max(float(np.max(np.abs(x))) for x in signals.values())
        gain = min(1.0, 0.98 / common_peak) if common_peak > 0 else 1.0
        rows = []
        for name, x in signals.items():
            level = rms(x)
            peak = float(np.max(np.abs(x)))
            rows.append({
                "Tín hiệu": name, "RMS": level, "Peak": peak,
                "Mean (DC)": float(np.mean(x)),
                "STD": float(np.std(x)),
                "Crest Factor": peak / level if level > 0 else np.nan,
            })
        metrics = pd.DataFrame(rows)
        if reference is not None:
            ratios = {name: reference_ratio_db(reference, x) for name, x in signals.items()}
            metrics["Tín hiệu/sai số (dB)"] = metrics["Tín hiệu"].map(ratios)
            change = ratios["Sau lọc"] - ratios["Trước lọc"]
            st.metric("Thay đổi tỷ số tín hiệu/sai số", f"{change:+.2f} dB")
            st.caption(
                "Trước lọc, chỉ số bằng SNR trộn. Sau lọc, sai số bao gồm nhiễu còn "
                "lại và méo/mất âm hữu ích; số lớn hơn nghĩa là gần tham chiếu hơn."
            )
        else:
            st.info(
                "Không có tham chiếu nên không tính SNR. "
                "RMS giảm chỉ cho thấy mức tín hiệu giảm, chưa chứng minh lọc tốt."
            )
        st.dataframe(metrics.round(6), width="stretch", hide_index=True)

        st.subheader("Nghe so sánh")
        st.caption(f"Các bản WAV dùng chung gain = {gain:.6f}; bảng và hình dùng dữ liệu trước gain.")
        audio_data = {}
        for column, (name, x) in zip(st.columns(len(signals)), signals.items()):
            audio_data[name] = wav_bytes(x, sr, gain)
            column.write(name)
            column.audio(audio_data[name], format="audio/wav")

        # 5. HÌNH DÙNG CHUNG TRỤC VÀ XUẤT MINH CHỨNG
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
        t = np.arange(len(before)) / sr
        for name, x in signals.items():
            axes[0].plot(t, x, linewidth=0.45, alpha=0.7, label=name)
            # Welch trung bình các phổ đoạn, giảm dao động của ước lượng PSD.
            f, psd = welch(x, fs=sr, nperseg=min(2048, len(x)), detrend=False)
            axes[1].plot(f, 10 * np.log10(np.maximum(psd, 1e-16)), label=name)
        axes[0].set(title="Dạng sóng trên cùng thang biên độ",
                    xlabel="Thời gian (s)", ylabel="Biên độ số")
        axes[1].set(title="Mật độ phổ công suất (Welch)",
                    xlabel="Tần số (Hz)", ylabel="PSD (dB re 1 đơn vị²/Hz)")
        f, response_db = response_curve(sr, family, coefficients)
        axes[2].plot(f, response_db, label="Đáp ứng hai chiều |H(f)|²")
        for edge in np.atleast_1d(cutoff):
            axes[2].axvline(edge, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set(title=f"{family} {kind} - đáp ứng hiệu dụng",
                    xlabel="Tần số (Hz)", ylabel="Độ lợi (dB)", ylim=(-100, 5))
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"{source_file.name} | SR={sr} Hz | cutoff={cutoff}", fontsize=10)
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=200)
        st.pyplot(fig)
        plt.close(fig)

        # Lưu tham số cùng bảng để thí nghiệm có thể tái lập.
        export = metrics.assign(
            Source=source_file.name, Noise=noise_name, Mode=mode,
            SR_Hz=sr, Duration_s=len(before) / sr, Family=family, Filter=kind,
            Cutoff_Hz=str(cutoff), Order_one_pass=single_order,
            Numtaps=numtaps if family == "FIR" else np.nan,
            Method="forward-backward", Target_SNR_dB=snr_target, Seed=seed,
            WAV_gain=gain,
        )
        st.download_button("Tải bảng kết quả CSV",
                           export.to_csv(index=False).encode("utf-8-sig"),
                           "noise_filter_results.csv", "text/csv", key="t4_csv")
        st.download_button("Tải hình PNG", png.getvalue(),
                           "noise_filter_comparison.png", "image/png", key="t4_png")
        st.download_button("Tải WAV sau lọc", audio_data["Sau lọc"],
                           "filtered_audio.wav", "audio/wav", key="t4_wav")
        st.caption(
            "Lọc hai chiều làm biên độ đáp ứng thành |H|². "
            "Mốc cắt Butterworth tương ứng khoảng −6 dB sau hai lượt. "
            "FIR firwin thường khoảng −12 dB ở mốc cắt sau hai lượt khi "
            "dải chuyển tiếp đủ tách biệt; hãy xem đáp ứng thực tế. "
            "Cùng tần số cắt không có nghĩa FIR và IIR có đáp ứng giống nhau. "
            "Nhiễu trùng dải âm hữu ích vẫn còn; cắt quá mạnh có thể làm mất họa âm."
        )
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Chưa thể xử lý: {exc}")


if __name__ == "__main__":
    st.set_page_config(page_title="6-STARS | Nhiễu và lọc", layout="wide")
    render_noise_filter_tab()
