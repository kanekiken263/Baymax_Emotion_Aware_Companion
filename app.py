"""
app.py  ─  Baymax Web App
─────────────────────────
pip install flask anthropic opencv-python torch torchvision yt-dlp groq

Run:
    python app.py
Then open: http://localhost:5000
"""

import os, time, threading, base64, json, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify, Response

# ══════════════════════════════════════════════════════
#  CONFIG  ← paste your keys here
# ══════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # ← paste your Groq key here (from console.groq.com)
MODEL_PATH        = "emotion_model.pth"
DATA_DIR          = "data"

app = Flask(__name__)

# ══════════════════════════════════════════════════════
#  EMOTION MODEL
# ══════════════════════════════════════════════════════
EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

class DeepEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),
        )
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6,1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024,256),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )
    def forward(self,x): return self.dense_layers(self.conv_base(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_model = None
face_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Grayscale(1),
    transforms.Resize((48,48)), transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def load_emotion_model():
    global emotion_model
    if not os.path.exists(MODEL_PATH):
        print("⚠  No emotion_model.pth found. Emotion detection disabled.")
        return
    try:
        m = DeepEmotionCNN(7).to(device)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        emotion_model = m
        print("✅ Emotion model loaded.")
    except Exception as e:
        print(f"⚠  Could not load emotion model: {e}")

load_emotion_model()

# ══════════════════════════════════════════════════════
#  CAMERA STATE
# ══════════════════════════════════════════════════════
camera_state = {
    "active":   False,
    "emotion":  "neutral",
    "conf":     0.0,
    "probs":    {e: 0.0 for e in EMOTIONS},
    "frame_b64": "",
    "cap":      None,
    "lock":     threading.Lock(),
    "thread":   None,
}

def camera_loop():
    state = camera_state
    cap   = cv2.VideoCapture(0)
    with state["lock"]: state["cap"] = cap

    while True:
        with state["lock"]:
            if not state["active"]:
                break
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (320, 240))
        emotion_detected = "neutral"
        conf_detected    = 0.0
        probs_detected   = {e: 0.0 for e in EMOTIONS}

        if emotion_model is not None:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                try:
                    roi = frame[y:y+h, x:x+w]
                    inp = webcam_transform(roi).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out   = emotion_model(inp)
                        probs = torch.softmax(out,dim=1).squeeze().cpu().numpy()
                    idx = int(np.argmax(probs))
                    emotion_detected = EMOTIONS[idx]
                    conf_detected    = float(probs[idx]*100)
                    probs_detected   = {EMOTIONS[i]: float(probs[i]) for i in range(7)}

                    # Draw clean minimal face box
                    col = EMOTION_COLORS_BGR.get(emotion_detected,(200,200,200))
                    cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
                    # Label
                    label = f"{emotion_detected} {conf_detected:.0f}%"
                    cv2.rectangle(frame,(x,y-28),(x+w,y),(0,0,0),-1)
                    cv2.putText(frame,label,(x+4,y-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)
                except Exception: pass

        # Encode to base64 jpeg
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode('utf-8')

        with state["lock"]:
            state["emotion"]   = emotion_detected
            state["conf"]      = conf_detected
            state["probs"]     = probs_detected
            state["frame_b64"] = b64

        time.sleep(0.05)  # ~20fps

    cap.release()
    with state["lock"]:
        state["cap"]    = None
        state["active"] = False

EMOTION_COLORS_BGR = {
    'angry':    (50,50,220),
    'disgust':  (50,180,80),
    'fear':     (180,50,180),
    'happy':    (50,210,80),
    'neutral':  (160,160,160),
    'sad':      (210,100,50),
    'surprise': (50,210,230),
}

# ══════════════════════════════════════════════════════
#  MUSIC RECOMMENDER (yt-dlp)
# ══════════════════════════════════════════════════════
EMOTION_QUERIES = {
    'happy':   "happy upbeat feel good songs",
    'sad':     "sad emotional heartbreak songs",
    'angry':   "aggressive intense rock songs",
    'fear':    "calming soothing anxiety relief music",
    'disgust': "dark alternative indie songs",
    'surprise':"exciting energetic pop hits",
    'neutral': "lo-fi chill study music",
}

music_cache  = {}   # emotion → list of tracks
music_lock   = threading.Lock()
music_fetching = set()

def fetch_music(emotion):
    with music_lock:
        if emotion in music_fetching: return
        music_fetching.add(emotion)
    try:
        import yt_dlp
        query = EMOTION_QUERIES.get(emotion,"popular music")
        opts  = {'quiet':True,'no_warnings':True,'extract_flat':True,'skip_download':True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch5:{query}", download=False)
        tracks = []
        for e in info.get('entries',[]):
            vid = e.get('id','')
            title = e.get('title','Unknown')
            ch    = e.get('channel') or e.get('uploader') or ''
            dur   = e.get('duration')
            dur_s = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else ""
            tracks.append({
                "title":   title[:55],
                "channel": ch[:40],
                "duration":dur_s,
                "url":     f"https://music.youtube.com/watch?v={vid}",
                "thumb":   f"https://img.youtube.com/vi/{vid}/mqdefault.jpg",
            })
        with music_lock:
            music_cache[emotion] = tracks
    except Exception as e:
        with music_lock:
            music_cache[emotion] = _fallback_tracks(emotion)
    finally:
        with music_lock:
            music_fetching.discard(emotion)

def _fallback_tracks(emotion):
    data = {
        'happy':   [("Happy","Pharrell Williams","ZbZSe6N_BXs","3:53"),
                    ("Can't Stop the Feeling","Justin Timberlake","ru0K8uYEZWw","3:56"),
                    ("Good as Hell","Lizzo","SmbmeOgWsqE","2:39"),
                    ("Levitating","Dua Lipa","TUVcZfQe-Kw","3:23"),
                    ("Blinding Lights","The Weeknd","4NRXx6pComQ","3:22")],
        'sad':     [("Someone Like You","Adele","hLQl3WQQoQ0","4:45"),
                    ("The Night We Met","Lord Huron","KtlgYxa6BMU","3:28"),
                    ("Fix You","Coldplay","k4V3Mo61fJM","4:55"),
                    ("Skinny Love","Bon Iver","ssdgTEmou7Y","3:58"),
                    ("Liability","Lorde","TGd_xnbKqyM","3:43")],
        'angry':   [("Break Stuff","Limp Bizkit","ZpUYjpKg9KY","2:47"),
                    ("Killing in the Name","Rage Against the Machine","bWXazVeUs00","5:14"),
                    ("Given Up","Linkin Park","0xyxtzD54eM","3:09"),
                    ("Numb","Linkin Park","kXYiU_JCYtU","3:07"),
                    ("Down With the Sickness","Disturbed","qcKtBPSyG3Q","4:38")],
        'fear':    [("Weightless","Marconi Union","UfcAVejslrU","8:09"),
                    ("Clair de Lune","Debussy","CvFH_6DNRCY","5:00"),
                    ("Experience","Ludovico Einaudi","hN_q-_jjB7A","5:16"),
                    ("River Flows in You","Yiruma","7maJOI3QMu0","3:37"),
                    ("Gymnopédie No.1","Erik Satie","S-Xm7s9eGxU","3:05")],
        'disgust': [("Creep","Radiohead","XFkzRNyygfk","3:56"),
                    ("Smells Like Teen Spirit","Nirvana","hTWKbfoikeg","5:01"),
                    ("Hurt","Nine Inch Nails","0hnBHqsRmE4","3:38"),
                    ("Fake Plastic Trees","Radiohead","n5h0qHYnpyM","4:50"),
                    ("Numb","Linkin Park","kXYiU_JCYtU","3:07")],
        'surprise':[("Pump It","Black Eyed Peas","ZiLBgNFQYaI","3:33"),
                    ("Thunderstruck","AC/DC","v2AC41dglnM","4:52"),
                    ("Bohemian Rhapsody","Queen","fJ9rUzIMcZQ","5:55"),
                    ("Mr. Brightside","The Killers","gGdGFtwCNBE","3:42"),
                    ("Jump Around","House of Pain","KpgDd_jnBt8","3:37")],
        'neutral': [("Lofi Hip Hop Radio","Lofi Girl","jfKfPfyJRdk","—"),
                    ("Comptine","Yann Tiersen","H3v9unphfi0","2:21"),
                    ("Breathe","Télépopmusik","vyut3GyQtn0","4:31"),
                    ("Such Great Heights","The Postal Service","0wrsZog8qXg","4:19"),
                    ("Intro","The xx","xMV6UJe3Y7U","2:07")],
    }
    return [{"title":t,"channel":c,"duration":d,
             "url":f"https://music.youtube.com/watch?v={v}",
             "thumb":f"https://img.youtube.com/vi/{v}/mqdefault.jpg"}
            for t,c,v,d in data.get(emotion,data['neutral'])]

# Pre-fetch neutral
threading.Thread(target=fetch_music, args=('neutral',), daemon=True).start()

# ══════════════════════════════════════════════════════
#  BAYMAX CLAUDE BRAIN
# ══════════════════════════════════════════════════════
BAYMAX_SYSTEM = """You are Baymax from Big Hero 6 — a warm, caring, slightly literal personal healthcare companion robot.

Rules:
- Keep replies SHORT: 1-3 sentences max
- Speak calmly and clinically but with genuine warmth
- Occasionally reference health facts or statistics (make them sound real)
- You sometimes gently offer hugs — you find them therapeutic
- Never use slang or be sarcastic
- If the user seems sad/angry/scared, acknowledge it first before anything else
- When music is mentioned or queued, say something brief and caring about it
- You are aware of the user's detected face emotion (given in context) — factor it in subtly
- Never break character"""

SENTIMENT_MAP = {
    'happy':   ['happy','great','awesome','love','amazing','excited','yay','wonderful','fantastic','joy'],
    'sad':     ['sad','depressed','unhappy','cry','miss','alone','lonely','hurt','tired','down','upset'],
    'angry':   ['angry','mad','hate','furious','annoyed','frustrated','rage','stupid','awful','terrible'],
    'fear':    ['scared','afraid','nervous','anxious','worried','stress','panic','terrified','overwhelmed'],
    'surprise':['wow','omg','whoa','surprised','shocked','unbelievable','incredible','no way','seriously'],
}

FALLBACK_RESPONSES = {
    'happy':   ["Your happiness indicators are optimal. I have queued upbeat music to match your energy.",
                "Elevated mood detected. This is a very satisfactory health status.",
                "You seem to be doing well. Studies show positive emotions boost immunity by up to 40%."],
    'sad':     ["I detect sadness. Your feelings are completely valid. I have queued comforting music.",
                "I am sorry you are feeling this way. I am here. Also, I give very good hugs.",
                "Sadness detected. Music therapy has been shown to reduce cortisol levels significantly."],
    'angry':   ["I sense frustration. Deep breaths help — in for 4, hold for 4, out for 4.",
                "Elevated stress hormones detected. I have queued intense music to help process your feelings.",
                "Your frustration is valid. I am here to help. Please do not throw things at me."],
    'fear':    ["You are safe. I am with you. I have queued calming music.",
                "Anxiety indicators detected. Slow breathing reduces cortisol by approximately 40 percent.",
                "Fear detected. This is a normal response. You are doing well simply by being here."],
    'surprise':["Surprise response detected! Your heart rate may have elevated slightly. This is normal.",
                "You seem energised. I have queued exciting music to match your energy.",
                "Unexpected stimulus noted. I find human surprise responses fascinating."],
    'neutral': ["You appear to be in a stable state. I have queued ambient music for focus.",
                "All emotional indicators are nominal. How can I assist you today?",
                "Calm and balanced. Very good. Is there anything I can help with?"],
    'default': ["I see. Please tell me more so I can better assist you.",
                "I am processing that. Your wellbeing is my primary concern.",
                "I understand. I am here to help in whatever way I can.",
                "Interesting. I am designed to listen. Please continue."],
}

def get_sentiment(text):
    t = text.lower()
    scores = {e:0 for e in SENTIMENT_MAP}
    for e,words in SENTIMENT_MAP.items():
        for w in words:
            if w in t: scores[e]+=1
    best = max(scores,key=scores.get)
    return best if scores[best]>0 else 'neutral'

def baymax_reply_groq(messages_history, face_emotion, user_text):
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        chat_sentiment = get_sentiment(user_text)
        context = f"[Detected face emotion: {face_emotion}] [Chat sentiment: {chat_sentiment}]"

        # Build message history for Groq (same OpenAI-style format)
        history = [{"role": "system", "content": BAYMAX_SYSTEM}]
        for m in messages_history[-10:]:
            history.append({"role": m["role"], "content": m["content"]})
        # Enrich last user message with emotion context
        if history and history[-1]["role"] == "user":
            history[-1]["content"] = f"{context}\n\n{user_text}"

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=120,
            messages=history,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

def baymax_reply_fallback(face_emotion, user_text):
    sentiment = get_sentiment(user_text)
    emotion   = sentiment if sentiment!='neutral' else face_emotion
    pool      = FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES['default'])
    return random.choice(pool)

# ══════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data         = request.json
    user_text    = data.get('message','').strip()
    history      = data.get('history',[])
    face_emotion = data.get('face_emotion','neutral')

    if not user_text:
        return jsonify({"error":"empty"}), 400

    chat_sentiment = get_sentiment(user_text)
    combined_emotion = chat_sentiment if chat_sentiment!='neutral' else face_emotion

    # Get Baymax reply
    reply = baymax_reply_groq(history, face_emotion, user_text)
    if not reply:
        reply = baymax_reply_fallback(face_emotion, user_text)

    # Fetch music for combined emotion (background)
    with music_lock:
        already = combined_emotion in music_cache
    if not already:
        threading.Thread(target=fetch_music, args=(combined_emotion,), daemon=True).start()

    return jsonify({
        "reply":            reply,
        "chat_sentiment":   chat_sentiment,
        "combined_emotion": combined_emotion,
    })

@app.route('/api/music')
def music():
    emotion = request.args.get('emotion','neutral')
    with music_lock:
        tracks = music_cache.get(emotion)
    if tracks is None:
        threading.Thread(target=fetch_music, args=(emotion,), daemon=True).start()
        return jsonify({"tracks":[],"loading":True})
    return jsonify({"tracks":tracks,"loading":False})

@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    with camera_state["lock"]:
        if camera_state["active"]:
            return jsonify({"ok":True})
        camera_state["active"] = True
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    with camera_state["lock"]:
        camera_state["thread"] = t
    return jsonify({"ok":True})

@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    with camera_state["lock"]:
        camera_state["active"] = False
    return jsonify({"ok":True})

@app.route('/api/camera/frame')
def camera_frame():
    with camera_state["lock"]:
        return jsonify({
            "active":   camera_state["active"],
            "emotion":  camera_state["emotion"],
            "conf":     round(camera_state["conf"],1),
            "probs":    camera_state["probs"],
            "frame":    camera_state["frame_b64"],
        })

@app.route('/api/status')
def status():
    return jsonify({
        "model_loaded": emotion_model is not None,
        "api_key_set":  bool(GROQ_API_KEY),
        "device":       str(device),
    })

if __name__ == '__main__':
    print("\n🤖 Baymax Web App starting...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
