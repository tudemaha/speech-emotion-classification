import librosa

def load(uploaded_files):
    audio_librosa = []
    for uploaded_file in uploaded_files:
        audio, _ = librosa.load(uploaded_file, sr = 16000)
        audio_librosa.append(audio)

    return audio_librosa