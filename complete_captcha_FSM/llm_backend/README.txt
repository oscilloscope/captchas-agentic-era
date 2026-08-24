Create Virtual Environment

python3.11 -m venv holo1_venv

---
Activate Virtual Environment

Linux/macOS:
source holo1_venv/bin/activate

Windows:
holo1_venv\Scripts\activate

---

Install Dependencies

pip install -r requirements.txt

---
Run the Application

python gui_agent_backed.py

With public ngrok tunnel:
python gui_agent_backed.py --ngrok-token YOUR_NGROK_TOKEN

---
First Run

- GUI Agent Model download: about 60GB (one-time, cached in ~/.cache/huggingface/)

---
API Endpoints

| Method | Endpoint  | Description              |
|--------|-----------|--------------------------|
| GET    | /         | Health check / status    |
| POST   | /generate | Run inference with image |

---
Example API Call

curl -X POST "http://localhost:8000/generate" \
-F "image=@screenshot.png" \
-F 'messages_json=[{"role":"user","content":[{"type":"image"},{"type":"text","text":"What do you see?"}]}]'

---