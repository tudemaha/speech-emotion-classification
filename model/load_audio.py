# import librosa module
import librosa

# function to load audio files
def load(uploaded_files):
    # prepare empty list to store audio
    audio_librosa = []
    # load the audio data from uploaded files then append it to the list
    for uploaded_file in uploaded_files:
        audio, _ = librosa.load(uploaded_file, sr = 22050)
        audio_librosa.append(audio)

    # return the list
    return audio_librosa