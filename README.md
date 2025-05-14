
````markdown
# 🗣️ Text-to-Speech (TTS) Pipeline Using NVIDIA NeMo

This project demonstrates a **Text-to-Speech (TTS) pipeline** using NVIDIA's **NeMo** framework. The notebook includes an end-to-end process: converting an audio file to text using **QuartzNet (ASR)** and then converting the text back to synthetic speech using **FastPitch + HiFi-GAN (TTS)**.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Code Workflow](#code-workflow)
- [Further Implementations](#further-implementations)
- [Real-World Use Cases](#real-world-use-cases)
- [Conclusion](#conclusion)
- [License](#license)

---

## ✅ Overview

This notebook walks you through:

- Loading and analyzing an audio file
- Transcribing it into text using **QuartzNet**
- Generating mel spectrograms using **FastPitch**
- Synthesizing human-like speech from spectrograms using **HiFi-GAN**
- Visualizing waveforms and spectrograms using **Librosa**

---

## 🛠 Technologies Used

| Tool           | Purpose                             |
|----------------|-------------------------------------|
| Python         | Programming language                |
| NVIDIA NeMo    | ASR and TTS modeling toolkit        |
| QuartzNet      | Speech-to-text (ASR)                |
| FastPitch      | Text-to-spectrogram (TTS)           |
| HiFi-GAN       | Spectrogram-to-audio (Vocoder)      |
| Librosa        | Audio processing and visualization  |
| IPython        | Audio playback in notebook          |
| Matplotlib     | Plotting waveforms and spectrograms |

---

## 🧩 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pylasandeep52/Automatic_speech_recognition.git
cd your-repository
````

### 2️⃣ Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv tts_env

# Activate (Linux/macOS)
source tts_env/bin/activate

# Activate (Windows)
tts_env\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install nemo_toolkit[all]
pip install librosa matplotlib
```

---

## 🧪 Usage Instructions

### 📁 Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook `TTS_Pipeline.ipynb` and run each cell in order.

### 🔁 Process Flow

1. **Download Audio Sample**

   ```python
   !wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
   ```

2. **Transcribe Audio with QuartzNet**

   ```python
   quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_quartznet15x5").cuda()
   transcription = quartznet.transcribe(["2086-149220-0033.wav"])[0]
   print("Transcribed Text:", transcription)
   ```

3. **Generate Speech from Text using FastPitch + HiFi-GAN**

   ```python
   spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(model_name="tts_en_fastpitch").cuda()
   vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_en_hifigan").cuda()

   def text_to_audio(text):
       tokens = spectrogram_generator.parse(text)
       spectrogram = spectrogram_generator.generate_spectrogram(tokens=tokens)
       audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
       return audio.to('cpu').detach().numpy()
   ```

4. **Playback Audio Output**

   ```python
   IPython.display.Audio(text_to_audio(transcription), rate=22050)
   ```

---

## 🔬 Code Workflow

### 📥 Audio Loading & Visualization

* **Waveform**

  ```python
  A_S, sr = librosa.load("2086-149220-0033.wav", sr=None)
  librosa.display.waveshow(A_S, sr=sr)
  ```

* **Spectrograms**

  * STFT Spectrogram

    ```python
    spec = np.abs(librosa.stft(A_S))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
    ```
  * Mel Spectrogram

    ```python
    mel_spec = librosa.feature.melspectrogram(y=A_S)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel')
    ```

---

## 🚀 Further Implementations

* ✅ Real-time TTS from microphone input
* ✅ Support multiple languages or dialects
* ✅ Voice cloning and style transfer
* ✅ Emotion-based speech synthesis
* ✅ Build a chatbot or voice assistant interface

---

## 🌍 Real-World Use Cases

| Use Case             | Description                            |
| -------------------- | -------------------------------------- |
| Virtual Assistants   | Voice-based query systems              |
| Accessibility Tools  | Helping visually impaired users        |
| Language Learning    | Pronunciation practice and feedback    |
| Audiobook Generation | Converting text into narrated content  |
| Call Center Bots     | Automate responses with natural speech |

---

## 🧾 Conclusion

This project illustrates how to build an end-to-end TTS system using NVIDIA NeMo. It combines powerful models—QuartzNet, FastPitch, and HiFi-GAN—to transcribe audio and synthesize natural speech. With clear visualizations and modular components, this project is a strong foundation for advanced speech-based applications like voice assistants, chatbots, and AI narrators.

Feel free to extend, tweak, or integrate this into larger AI systems!

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

```

---

Let me know if you want me to include GitHub badges, links to datasets, or help you generate a LICENSE file too.
```
