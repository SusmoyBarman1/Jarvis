import speech_recognition as sr

r = sr.Recognizer()

harvard = sr.AudioFile('harvard.wav')

with harvard as source:
    audio = r.record(source)

#print(type(audio))

#text = r.recognize_google(audio)
#print(text)

# Capturing Segments With offset and duration
# offset keyword argument represents the number of seconds from the beginning of the file to ignore before starting to record.
# duration keyword argument that stops the recording after a specified number of seconds.
with harvard as source:
    audio = r.record(source, offset=4, duration=3)

#print(type(audio))

text = r.recognize_google(audio)
print()
print(text)