import speech_recognition as sr

r = sr.Recognizer()

jack = sr.AudioFile('jackhammer.wav')

with jack as source:
    audio = r.record(source)

text = r.recognize_google(audio, show_all=True)
print(text)