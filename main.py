import tkinter as tk
from time import perf_counter
import tkvideo
from PIL import Image,ImageTk
import cv2
import imutils
import tkvideo
import os
import re
import speech_recognition as sr
from unicodedata import normalize
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, format_sentences, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME, ROOT_PATH
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

actions = get_actions(DATA_PATH)
model_path = os.path.join(MODELS_PATH, MODEL_NAME)
model = load_model(model_path)
sentence = []
kp_sequence=[]
count_frame = 0
repe_sent = 1
time_last_written=0
holistic_model = Holistic()

dir = 'C:\\Users\\Usuario\\Desktop\\Prototipo_v2\\palabras'
listaVid = []
with os.scandir(dir) as ficheros:
     for fichero in ficheros:
            listaVid.append(fichero.name.replace('.mp4',''))

# create custom class based on tkvideo
class Player(tkvideo.tkvideo):
    # overload load()
    def load(self, path, label, loop):
        # call the original load()
        super().load(path, label, loop)
        # video ended, signal the application via virtual event
        label.event_generate("<<End>>")


def show_message(sentence):
    lblPalabras = tk.Label(main,text=sentence,bg="#0900c2",fg="white",width=55,height=3,font=("new academy",20))
    lblPalabras.place(x=108,y=936)
    main.after(3000, lblPalabras.destroy)

# function to play a video in the playlist
def play():
    global playlist
    try:
        # get next video from the playlist
        video = next(playlist)
        print(f"{video=}")
        # play the video
        player = Player(video, lblVid,size=(570,715))
        player.play()
    except Exception as ex:
        # no more video to play
        lblVid.configure(image="")
        print(f"{ex=}")

def mostrarWebcam():
    global kp_sequence
    global count_frame
    global repe_sent 
    global sentence
    global time_last_written
    threshold = 0.7
    if cap is not None:
        ret,frame = cap.read()
        if ret == True:
            frame=imutils.resize(frame,height=715)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            frame,results = mediapipe_detection(frame,holistic_model)
            kp_sequence.append(extract_keypoints(results))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
                
            else:
                if count_frame >= MIN_LENGTH_FRAMES:
                    r = np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0)
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]
                    p = res[np.argmax(res)]

                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                        sentence, repe_sent = format_sentences(sent, sentence, repe_sent)
                        
                    count_frame = 0
                    kp_sequence = []

            #frame = cv2.rectangle(frame, (0,0), (640, 35), (245, 117, 16), -1)
            #frame = cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            
            save_txt('outputs/sentences.txt', '\n'.join(sentence))
            draw_keypoints(frame, results)
        
        

        framex = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im   = Image.fromarray(framex)
        img = ImageTk.PhotoImage(image = im)

        lblWebcam.configure(image=img)
        lblWebcam.image = img           
        lblWebcam.after(50, mostrarWebcam)

    show_message(sentence)

def iniciar():
    global cap
    cap = cv2.VideoCapture(0)
    mostrarWebcam()

def audioTexto():
    global playlist
    playlist=[]
    colavid=[]
    listener = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        print("Escuchando...")
        audio = listener.listen(source)
        rec = listener.recognize_whisper(audio, language="spanish")
        trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
        rec = normalize('NFKC', normalize('NFKD', rec).translate(trans_tab))
        rec = re.sub(r'[.]', '', rec)
        palabras = rec.split(',')
    try:
        for palabra in palabras:
            for vid in listaVid:
                if vid in palabra:
                    colavid.append(vid+'.mp4')
        #return colavid
        # create the playlist
        playlist = iter(colavid)
        # call play() upon receiving virtual event "<<End>>"
        lblVid.bind("<<End>>", lambda e: play())
        # play the first video
        play()

    except sr.UnknownValueError:
        print("No se entiende")
    except sr.RequestError as e:
        print("error")

def textoVid(*args):
        global playlist
        playlist=[]
        colavid=[]
        inp = inputTxt.get(1.0, "end-1c")
        trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
        rec = normalize('NFKC', normalize('NFKD', inp).translate(trans_tab))
        rec = re.sub(r'[.]', '', rec)
        palabras = rec.split(',')
        try:
            for palabra in palabras:
                for vid in listaVid:
                    if vid == palabra:
                        colavid.append(vid+'.mp4')
        #return colavid
        # create the playlist
            playlist = iter(colavid)
        # call play() upon receiving virtual event "<<End>>"
            lblVid.bind("<<End>>", lambda e: play())
        # play the first video
            play()
            

        except sr.UnknownValueError:
            print("No se entiende")
        except sr.RequestError as e:
            print("error")
    

cap = None
main = tk.Tk()
main.geometry("1920x1080")
main.title("Traductor LsCh")
main.resizable(width=False,height=False)
fondo = tk.PhotoImage(file="prototipo_v2.png")
fondo1 = tk.Label(main,image=fondo).place(x=0,y=0,relwidth=1,relheight=1)

#Botones

btnWebcam = tk.Button(main,text="WEBCAM",bg="#f25781",fg="#ffffff",relief="flat",cursor="hand2",height=2,width=12,font=("new academy",24,"bold"),command=iniciar)
btnWebcam.place(x=882.3,y=76.6)

btnHablar = tk.Button(main,text="HABLAR",bg="#f25781",fg="#ffffff",relief="flat",cursor="hand2",height=2,width=12,font=("new academy",24,"bold"),command=audioTexto)
btnHablar.place(x=1541.3,y=76.6)

#Labels

lblWebcam = tk.Label(main,bg="black")
lblWebcam.place(x=108,y=190)

lblVid = tk.Label(main,bg="black")
lblVid.place(x=1259,y=190)

# lblPalabras = tk.Label(main,bg="#0900c2",fg="white",width=55,height=3,font=("new academy",20))
# lblPalabras.place(x=108,y=936)

#txt

inputTxt = tk.Text(main,bg="#0900c2",fg="white",relief="flat",height=3,width=27,font=("new academy",20))
inputTxt.place(x=1259,y=931)
inputTxt.bind("<Return>",textoVid)

main.state("zoomed")
main.mainloop()