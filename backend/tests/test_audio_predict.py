"""Run with: python -m unittest discover -s backend/tests -v."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio import audio_predict as audio
from app import app
import fusion


class AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "sample.wav"
        # Stereo 8 kHz source verifies FFmpeg resampling and mono conversion.
        samples = (np.sin(np.arange(8000 * 9) * 2 * np.pi * 220 / 8000) * 3000).astype('<i2')
        with wave.open(str(cls.path), 'wb') as output:
            output.setparams((2, 2, 8000, 0, 'NONE', 'not compressed'))
            output.writeframes(np.column_stack((samples, samples)).tobytes())
        cls.video_path = Path(cls.directory.name) / 'recording.mp4'
        subprocess.run([
            audio.imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-v', 'error',
            '-f', 'lavfi', '-i', 'color=c=black:s=32x32:r=1:d=9',
            '-i', str(cls.path), '-c:v', 'libx264', '-c:a', 'aac',
            '-shortest', str(cls.video_path),
        ], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_decode(self):
        signal = audio.load_audio(self.path)
        self.assertEqual(signal.shape, (16000 * 9,))
        self.assertEqual(signal.dtype, np.float32)

    def test_invalid_media_and_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            audio.predict_audio_emotion(Path(self.directory.name) / 'missing.wav')
        invalid = Path(self.directory.name) / 'invalid.mp4'
        invalid.write_bytes(b'not media')
        with self.assertRaisesRegex(ValueError, 'Audio extraction failed'):
            audio.predict_audio_emotion(invalid)

    def test_missing_checkpoint(self):
        with patch.object(audio, 'MODEL_PATH', Path(self.directory.name)):
            with self.assertRaisesRegex(FileNotFoundError, 'config.json'):
                audio._load_model.__wrapped__()

    def test_duration_weighting_and_config_order(self):
        import torch
        labels = ['sad', 'neutral', 'happy', 'fear', 'disgust', 'angry']
        model = unittest.mock.Mock()
        model.config.conv_kernel = [10]
        model.config.conv_stride = [5]
        model.side_effect = [types.SimpleNamespace(logits=torch.tensor([[10., 0, 0, 0, 0, 0]])),
                             types.SimpleNamespace(logits=torch.tensor([[0., 0, 0, 0, 0, 10]]))]
        extractor = unittest.mock.Mock(return_value={})
        with patch.object(audio, '_load_model', return_value=(model, extractor, labels)):
            result = audio.predict_audio_emotion(self.path)
        self.assertEqual(result['emotion'], 'sad')
        self.assertAlmostEqual(result['confidence'], 88.87, delta=.02)
        self.assertEqual(result['segments'], 2)

    def test_prediction_failure(self):
        with patch.object(audio, '_load_model', side_effect=RuntimeError('broken checkpoint')):
            with self.assertRaisesRegex(RuntimeError, 'Wav2Vec2 audio prediction failed'):
                audio.predict_audio_emotion(self.path)

    def test_real_inference_and_flask_fusion(self):
        result = audio.predict_audio_emotion(self.path)
        self.assertEqual(result['segments'], 2)
        self.assertEqual(set(result['probabilities']), audio.EMOTION_CLASSES)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 100, delta=.05)
        self.assertEqual(result['confidence'], result['probabilities'][result['emotion']])
        # Run the real Flask route and fusion with real audio inference, isolating
        # the unrelated video, Whisper and text models.
        video = types.ModuleType('video.video_predict')
        video.predict_video_emotion = lambda path: result
        speech = types.ModuleType('speech.whisper_predict')
        speech.transcribe_video = lambda path: {'text': 'A clear answer.'}
        text = types.ModuleType('text.text_predict')
        text.predict_answer_quality = lambda **kwargs: {'label': 'Good', 'probabilities': {'Poor': 0, 'Average': 0, 'Good': 100}}
        with patch.dict(sys.modules, {'video.video_predict': video, 'speech.whisper_predict': speech, 'text.text_predict': text}):
            response = app.test_client().post('/analyze', data={
                'question': 'Tell me about yourself',
                'video': (io.BytesIO(self.video_path.read_bytes()), 'recording.mp4'),
            }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body['audio']['segments'], 2)
        expected = (.4 * 100
                    + .3 * fusion._expected_score(result['probabilities'], fusion.EMOTION_SCORE_VALUES, 'Video')
                    + .3 * fusion._expected_score(body['audio']['probabilities'], fusion.EMOTION_SCORE_VALUES, 'Audio'))
        self.assertAlmostEqual(body['final']['score'], expected)

    def test_endpoint_reports_audio_failure(self):
        video = types.ModuleType('video.video_predict')
        video.predict_video_emotion = lambda path: {}
        with patch.dict(sys.modules, {'video.video_predict': video}):
            response = app.test_client().post('/analyze', data={
                'question': 'Question', 'video': (io.BytesIO(b'invalid'), 'bad.mp4'),
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn('Audio extraction failed', response.get_json()['details'])

    def test_cli_missing_input(self):
        completed = subprocess.run([sys.executable, audio.__file__, str(Path(self.directory.name) / 'missing.wav')], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        self.assertIn('Input file not found', json.loads(completed.stderr)['error'])


if __name__ == '__main__':
    unittest.main()
