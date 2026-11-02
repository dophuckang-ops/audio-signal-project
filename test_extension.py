import io
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import app_master as app
from streamlit.testing.v1 import AppTest
def wav(y, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="DOUBLE")
    return buf.getvalue()


class SignalTests(unittest.TestCase):
    def test_exact_slice_and_metadata(self):
        y = np.arange(1000, dtype=float) / 1000
        selected, meta = app.select_samples(y, 1000, 101, 601)
        np.testing.assert_array_equal(selected, y[101:601])
        self.assertEqual(meta['Start (s)'], .101)
        self.assertEqual(meta['End (s)'], .601)
        self.assertEqual(meta['Samples'], 500)
        self.assertEqual(meta['Source duration (s)'], 1)

    def test_invalid_intervals(self):
        for start, end in [(-1, 100), (50, 50), (60, 50), (0, 1001), (0, 31), (.1, 100)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                app.select_samples(np.ones(1000), 1000, start, end)

    def test_whole_file_backward_compatibility(self):
        y = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
        full, meta = app.select_samples(y, 16000)
        np.testing.assert_array_equal(full, y)
        self.assertEqual(app.time_features(full, 16000), app.time_features(y, 16000))
        self.assertEqual(meta['Duration (s)'], 1)

    def test_silence_changes_rms_not_gain(self):
        tone = .4 * np.sin(2 * np.pi * 400 * np.arange(16000) / 16000)
        y = np.r_[np.zeros(16000), tone]
        selected, _ = app.select_samples(y, 16000, 16000, 32000)
        self.assertAlmostEqual(app.time_features(selected, 16000)['RMS'], .4 / np.sqrt(2), places=12)
        self.assertAlmostEqual(app.time_features(y, 16000)['RMS'], .2, places=12)

    def test_f0_and_fft_use_selected_tone(self):
        t = np.arange(16000) / 16000
        y = np.r_[.3 * np.sin(2*np.pi*220*t), .3 * np.sin(2*np.pi*440*t)]
        data = wav(y)
        for start, end, hz in [(0,16000,220), (16000,32000,440)]:
            with self.subTest(hz=hz):
                result = app.estimate_f0(data, start_sample=start, end_sample=end)
                self.assertAlmostEqual(result['f0_hz'], hz, delta=3)
                self.assertAlmostEqual(app.fft_analysis(y[start:end],16000)['dominant_frequency'], hz, delta=1)
        short = app.estimate_f0(data, start_sample=0, end_sample=100)
        self.assertTrue(np.isnan(short['f0_hz']))

    def test_stft_stationary_tone_and_window_resolution(self):
        for sr in (16000, 44100, 48000):
            y = .3 * np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
            for window in (20,40,80):
                r = app.spectrogram_analysis(y,sr,window)
                peak = r['frequencies'][np.argmax(r['db'],axis=0)]
                np.testing.assert_allclose(peak, 1000, atol=r['bin_spacing_hz'])
                self.assertAlmostEqual(np.max(r['db']),0)
                self.assertTrue(np.min(r['db']) >= -80)
                self.assertAlmostEqual(r['bin_spacing_hz'],1000/window,places=8)

    def test_time_localization_and_coordinates(self):
        sr=16000
        y=np.zeros(sr); y[8000]=1
        r=app.spectrogram_analysis(y,sr,20)
        active=np.flatnonzero(np.max(r['db'],axis=0)>-3)
        self.assertTrue(np.all(np.abs(r['times'][active]-.5)<.021))
        fig=app.spectrogram_figure(r,sr,3.0,5000)
        self.assertGreater(fig.axes[0].get_xlim()[0],3)
        plt.close(fig)

    def test_silence_dc_short_and_single_frame(self):
        for level in (0.,.3):
            r=app.spectrogram_analysis(np.full(640,level),16000,40)
            self.assertTrue(r['silent'])
            self.assertEqual(r['db'].shape,(321,1))
            self.assertTrue(np.all(r['db']==-80))
            for limit in (0,8000):
                fig=app.spectrogram_figure(r,16000,0,limit)
                fig.canvas.draw()
                plt.close(fig)
        with self.assertRaises(ValueError):
            app.spectrogram_analysis(np.zeros(100),16000,40)

    def test_long_signal_memory_bound(self):
        r=app.spectrogram_analysis(np.zeros(16000*180),16000,80)
        self.assertTrue(r['adaptive_hop'])
        self.assertLessEqual(r['db'].size,500000)
        self.assertLessEqual(r['db'].shape[1],1200)
        self.assertLessEqual(r['last_frame_end_s'],180)

    def test_csv_and_listen_preserve_signal(self):
        y=np.linspace(-.5,.5,1000)
        segment,meta=app.select_samples(y,1000,200,600)
        encoded=app.segment_wav(segment,1000)
        decoded,sr=sf.read(io.BytesIO(encoded))
        np.testing.assert_allclose(segment,decoded,atol=3e-8)
        self.assertEqual(sr,1000)
        frame=app.with_segment_metadata(pd.DataFrame({'value':[1,2]}),meta)
        back=pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        self.assertTrue((back['Start sample (inclusive)']==200).all())
        self.assertTrue((back['End sample (exclusive)']==600).all())


BASE = f'''
import io,sys
sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})
import numpy as np
import soundfile as sf
import app_master as m
from unittest.mock import patch
class Upload:
    name="synthetic.wav"
    def getvalue(self):
        n=st.session_state.get("test_length",32000)
        freq=st.session_state.get("test_frequency",440)
        key=f"test_wav_{{n}}_{{freq}}"
        if key in st.session_state:
            return st.session_state[key]
        y=.3*np.sin(2*np.pi*freq*np.arange(n)/16000)
        buf=io.BytesIO();sf.write(buf,y,16000,format="WAV",subtype="DOUBLE")
        st.session_state[key]=buf.getvalue()
        return st.session_state[key]
st=m.st
'''


class UITests(unittest.TestCase):
    def make(self,tab):
        code=BASE+f'''
with patch.object(m.st,"file_uploader",side_effect=lambda *a,**kw: Upload() if kw.get("key")=={tab!r} else None):
    m.main()
'''
        result=AppTest.from_string(code,default_timeout=90).run()
        self.clean(result)
        return result

    def clean(self,at):
        self.assertFalse(at.exception, str(at.exception))
        self.assertFalse(at.error, str(at.error))

    def submit(self,at):
        next(b for b in at.button if b.label=='Phân tích đoạn này').click().run()
        self.clean(at)

    def test_tab1_submit_crop_reset_and_invalid(self):
        at=self.make('single_wav')
        self.assertFalse(at.dataframe)
        self.submit(at)
        self.assertTrue(at.dataframe)
        at.checkbox[0].uncheck()
        next(x for x in at.slider if "_range_" in x.key).set_value((.5,1.5))
        self.submit(at)
        self.assertEqual(at.session_state['tab1_applied_segment']['applied']['start'],8000)
        self.assertTrue(any(m.value=='16,000' and m.label=='Số mẫu' for m in at.metric))
        next(x for x in at.slider if "_range_" in x.key).set_value((.5,.5))
        self.submit(at)
        self.assertTrue(at.warning)
        self.assertFalse(at.dataframe)
        at.session_state['test_frequency']=660
        at.run();self.clean(at)
        self.assertFalse(at.dataframe)
        self.assertIsNone(at.session_state['tab1_applied_segment']['applied'])

    def test_tab3_spectrogram_crop_window_and_zero_limit(self):
        at=self.make('tab3_fft_uploader')
        self.assertFalse(at.dataframe)
        at.checkbox[0].uncheck()
        next(x for x in at.slider if "_range_" in x.key).set_value((.5,1.5))
        at.select_slider[0].set_value(80)
        self.submit(at)
        keys={x.proto.id.rsplit('-',1)[-1] for x in at.get('download_button')}
        self.assertTrue({'tab3_spectrogram_png','tab3_settings_csv','tab3_fft_png'}<=keys)
        self.assertEqual(at.session_state['tab3_applied_segment']['applied']['window_ms'],80)
        self.assertEqual(at.session_state['tab3_applied_segment']['applied']['start'],8000)
        self.assertEqual(at.session_state['tab3_applied_segment']['applied']['end'],24000)
        at.slider(key='tab3_frequency_limit_16000').set_value(0.).run()
        self.clean(at)
        self.assertTrue(at.dataframe)
        self.assertTrue(any(m.label=='F0 tổng hợp' for m in at.metric))

    def test_short_and_silent_ui(self):
        at=self.make('tab3_fft_uploader')
        at.session_state['test_length']=100
        at.run();self.submit(at)
        self.assertTrue(any('ngắn hơn cửa sổ' in x.value for x in at.info))
        self.assertTrue(any('không' in x.value.lower() for x in at.warning))
        at.session_state['test_length']=32000
        at.session_state['test_frequency']=0
        at.run();self.submit(at)
        self.assertTrue(any('chỉ có DC' in x.value for x in at.info))
        self.assertFalse(any(m.label=='Nốt nhạc gần nhất' for m in at.metric))

    def test_other_tabs_still_render(self):
        blank=AppTest.from_string(BASE+'m.main()',default_timeout=90).run()
        self.clean(blank)
        self.assertEqual(len(blank.tabs),4)
        at=self.make('t4_source')
        self.assertTrue(at.dataframe)
        code=BASE+'''
with patch.object(m.st,"file_uploader",side_effect=lambda *a,**kw: [Upload()] if kw.get("key")=="tab2_file_uploader" else None):
    m.main()
'''
        at=AppTest.from_string(code,default_timeout=90).run()
        self.clean(at)
        self.assertTrue(any('F0 (Hz)' in df.value.columns for df in at.dataframe))


if __name__=='__main__':
    unittest.main(verbosity=2)
