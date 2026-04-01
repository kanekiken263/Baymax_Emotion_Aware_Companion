# 🤖 Baymax — Emotion-Aware Music Companion

> *"Hello. I am Baymax, your personal healthcare companion."*

A full-stack AI application that combines **real-time facial emotion detection**, an **intelligent Baymax chatbot**, and **live music recommendations** — all in a clean, browser-based interface.

---

## What It Does

Baymax watches your face and listens to what you say. Based on how you're feeling — whether detected through your webcam or the words you type — he recommends music that matches your mood and responds with warmth, care, and the occasional health statistic.

**Face says sad + you type "I'm exhausted"?**  
Baymax acknowledges it, queues Adele, and offers a hug.

**You type "this is amazing!!" while looking happy?**  
Happy Vibes playlist. Immediately.

---

## How It Works

### Emotion Detection
A custom **DeepEmotionCNN** (built with PyTorch) processes your webcam feed in real time, classifying your facial expression into one of 7 emotions: `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`, `neutral`. The model is trained on the FER-2013 dataset with data augmentation and achieves strong accuracy across all 7 classes.

### Baymax AI Brain
Baymax is powered by the **Groq API** with a carefully crafted system prompt that keeps him fully in character — calm, caring, slightly literal, and always health-focused. A rule-based fallback is included for offline use or when no API key is provided.

### Smart Music Engine
Chat sentiment and face emotion are **combined** to determine the mood signal. Chat wins if it carries a strong emotional keyword; otherwise the face emotion drives the selection. Music is fetched live from **YouTube Music** via `yt-dlp` — no API key, no billing, completely free.

### Web Interface
A **Flask** backend serves a clean, responsive web UI. Everything runs locally — just open `http://localhost:5000`.

---

## Interface

| Section | Description |
|---|---|
| **Chat panel** | Talk to Baymax. He replies in character with typing animation and emotion tags |
| **Camera bubble** | Floating draggable circle showing your live webcam feed with emotion label |
| **Emotion bar** | Live probability bars for all 7 emotions, updates every 120ms |
| **Music panel** | 5 YouTube Music tracks with thumbnails, auto-updates with your mood |

---

## Project Structure

```
baymax_app/
├── app.py                  # Flask backend, emotion model, music engine, Claude API
├── emotion_model.pth       # Pre-trained DeepEmotionCNN weights
├── data/
│   ├── train/              # Training data (FER-2013 format)
│   └── test/               # Test data
└── templates/
    └── index.html          # Full frontend (HTML + CSS + JS, single file)
```

---

## Setup & Installation

### 1. Install dependencies
```bash
pip install flask torch torchvision opencv-python yt-dlp anthropic
```

### 2. Add your Anthropic API key *(optional but recommended)*
Open `app.py` and paste your key on line 18:
```python
ANTHROPIC_API_KEY = "sk-ant-..."
```
Get a free key at [console.anthropic.com](https://console.anthropic.com). Without it, Baymax runs in rule-based mode — still fully in character.

### 3. Place your trained model
Make sure `emotion_model.pth` is in the `baymax_app/` folder.  
If it doesn't exist, the app still runs — emotion detection will be disabled until a model is present.

### 4. Run
```bash
cd baymax_app
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## Controls

| Action | How |
|---|---|
| Chat with Baymax | Type in the input box and press Enter |
| Enable camera | Click **Enable Camera** in the top-right |
| Move camera bubble | Drag it anywhere on screen |
| Refresh music | Click **Refresh songs** in the music panel |
| Suggested prompts | Click any chip on the welcome screen |

---

## Model Architecture

```
DeepEmotionCNN
├── Block 1: Conv2d(1→64) × 2 + BN + ReLU + MaxPool + Dropout
├── Block 2: Conv2d(64→128) × 2 + BN + ReLU + MaxPool + Dropout
├── Block 3: Conv2d(128→256) × 2 + BN + ReLU + MaxPool + Dropout
└── Dense:   Linear(9216→1024) → Linear(1024→256) → Linear(256→7)
```

- Input: 48×48 grayscale face crops
- Output: softmax over 7 emotion classes
- Training: 25 epochs, Adam optimizer, label smoothing, StepLR scheduler
- Augmentation: horizontal flip, rotation, color jitter, affine translate

---

## Tech Stack

| Layer | Technology |
|---|---|
| Emotion Model | PyTorch, OpenCV, Haar Cascades |
| AI Chatbot | Anthropic Claude API (`claude-sonnet`) |
| Music Search | `yt-dlp` (YouTube, no API key needed) |
| Backend | Python, Flask |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## Features at a Glance

-  **7-class real-time emotion detection** from webcam
-  **Groq-powered Baymax** — stays fully in character
-  **Live YouTube Music search** — no API key, no cost
-  **Combined mood signal** — face emotion + chat sentiment
-  **Draggable camera bubble** — minimal, non-intrusive
-  **Live emotion probability bars** — all 7 emotions shown
-  **Offline fallback** — rule-based Baymax + curated playlists
-  **Browser-based** — no desktop window clutter

---

## License

This project was built for educational and demonstration purposes.  
Baymax is a character owned by Disney / Marvel. This is a fan project.

