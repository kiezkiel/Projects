import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import os

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')

for voice in voices:
    if len(voice.languages) > 1 and voice.languages[1] == 'ja_JP':
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)

def talk(text):
    engine.say(text)
    engine.runAndWait()

def x64_Native(command):
    vs_path = r'"D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"'
    architecture = "x64"
    full_command = f'cmd.exe /k "{vs_path} {architecture} && {command}"'
    os.system(full_command)

def take_command():
    command = ''
    with sr.Microphone() as source:
        print('listening...')
        voice = listener.listen(source)
        try:
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'kobayashi' in command:
                command = command.replace('kobayashi', '')
                print(command)
        except sr.UnknownValueError:
            print("Sorry, I did not get that")
        except sr.RequestError as e:
            print("Sorry, my speech service is down")
    return command

def run_alexa():
    command = take_command()
    print(command)
    if 'play' in command:
        music_type = command.replace('play', '').strip()
        if music_type:
            talk('playing ' + music_type + ' music')
            pywhatkit.playonyt(music_type + ' music')
        else:
            talk('Please specify the type of music you want to play.')
    elif 'brave ' in command:
        os.system('Start brave https://9animetv.to/watch/mysterious-disappearances-19126?ep=124294')
        talk('yes goshojin sama opening brave browser')
    elif 'open virtual machine' in command:
        os.system(r'"D:\Program Files\Oracle\VirtualBox\VirtualBox.exe"')
        talk('oke goshojin')
    elif 'open waves' in command:
        os.system(r'"D:\Wuthering Waves\launcher.exe"')
        talk('Yes goshojin sama opening ea')
    elif 'malware' in command:
        compile_command = r'cl "O:\KobayashiMalware\KobayashiMalware.cpp"'
        x64_Native(compile_command)
        talk('Compiling KobayashiMalware')
    elif 'cleaner' in command:  # Highlighted change
        cleaner_command = r'"J:\KobayashiMalware\cleaner.bat"'  # Highlighted change
        x64_Native(cleaner_command)  # Highlighted change
        talk('Running Malware Cleaner')  # Highlighted change
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'who the heck is' in command:
        person = command.replace('who the heck is', '')
        try:
            info = wikipedia.summary(person, 1)
            print(info)
            talk(info)
        except wikipedia.exceptions.PageError:
            talk("Sorry, I couldn't find any information on " + person)
    elif 'date' in command:
        date = datetime.date.today().strftime('%B %d, %Y')
        talk('Today\'s date is ' + date)

while True:
    run_alexa()
