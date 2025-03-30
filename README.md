
# 🥊 Rocky AI - Real-Time Boxing Coach on Raspberry Pi

Rocky AI is a **real-time pose estimation app** built for Raspberry Pi 5 with the **Raspberry Pi AI Camera (IMX500)**. It detects body movements during shadowboxing, tracks keypoints like wrists and ankles, and delivers motivational voiceovers in **Rocky Balboa's style**, powered by **OpenAI** and **ElevenLabs**.



https://github.com/user-attachments/assets/d4919ba5-95ff-4118-863a-e3ce41e53d1f


---

## 📦 Features

- 🎯 **Pose Estimation** using the IMX500's onboard neural network
- 🤖 **Real-time feedback** based on wrist & ankle movement and speed
- 🧠 **AI commentary** powered by OpenAI (`gpt-3.5-turbo`)
- 🎙️ **Text-to-speech voice feedback** via ElevenLabs (Rocky's voice!)
- 🥋 **Custom object detection** (punching bag) with Roboflow
- 📊 Automatic metrics every 5 seconds: speed, power, hand/leg movement
- 🔴 Detects inactivity and ends session if no movement for 3 seconds

---

## 🧠 Tech Stack

| Tool            | Purpose                               |
|-----------------|----------------------------------------|
| **Raspberry Pi 5** | Host device for real-time processing |
| **Pi AI Camera (IMX500)** | Edge pose estimation inference |
| **Picamera2**   | Video streaming and frame access      |
| **OpenAI API**  | Rocky-style feedback generation       |
| **ElevenLabs API** | Realistic voice synthesis           |
| **Roboflow API** | Punching bag detection via inference |
| **OpenCV**      | Frame processing and line drawing     |
| **NumPy**       | Keypoint math and analysis            |

---

## 🛠️ Requirements

- Raspberry Pi 5
- Raspberry Pi AI Camera (Sony IMX500)
- `libcamera` and `picamera2` setup
- Python 3.9+
- Network access (for APIs)
- VLC media player (for TTS playback)

---

## 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/yitaim18/rocky-ai.git
cd rocky-ai

# Install dependencies
pip install -r requirements.txt

# For Picamera2
sudo apt install -y python3-picamera2

# Optional: if VLC isn't installed already
sudo apt install -y vlc

# Setup OpenAI, ElevenLabs, Roboflow keys in your script
export OPENAI_API_KEY="your_openai_key"
export ELEVEN_API_KEY="your_elevenlabs_key"
export ROBOTFLOW_API_KEY="your_roboflow_key"

# Run the app
python imx500_final_rocky.py
```

---

## 📂 File Structure

```
rocky-ai/
│
├── imx500_final_rocky.py     # Main logic for pose, detection, AI feedback
├── output.mp3                # Temporary audio playback file
└── README.md                 # You're here!
```

---

## 📈 Metrics Explained

- **Speed** — based on average wrist speed
- **Power** — combination of wrist & ankle speed
- **Hand Movement** — total wrist distance
- **Leg Movement** — total ankle distance

Each metric is normalized to a scale of 0–10.

---

## 🎤 Example Output

```txt
[INFO] Boxing pose app started.
💬 Rocky says: Hands looking sharp, keep pushing those jabs champ!
```

---

## 📸 Hardware Acceleration

- Pose estimation runs on the **Sony IMX500** chip directly
- Bag detection uses cloud inference from **Roboflow**

---

## 🤖 API Usage Notes

- **OpenAI** is used to create motivational messages using GPT
- **ElevenLabs** converts these messages to speech (voice ID customizable)
- **Roboflow** detects the punching bag's position to trigger responses

---

## ⚠️ Disclaimers

- This is a hobby/experimental project, not a certified fitness tool.
- Always warm up and box in a safe environment.


