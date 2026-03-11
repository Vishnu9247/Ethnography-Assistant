from audio_transcriber import record_audio, transcribe

audio_path = record_audio('./temp')

text = transcribe(audio_path = audio_path)
file_path = 'D:/git repos/Ethnography Assistant/temp/temp.txt'

with open(file_path, 'w') as file:
    file.write(text)
print('file created')