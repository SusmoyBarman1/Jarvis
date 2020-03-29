import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

microPhoneNames = sr.Microphone.list_microphone_names()
#print(microPhoneNames)

with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

text = r.recognize_google(audio)
print()
print(text)