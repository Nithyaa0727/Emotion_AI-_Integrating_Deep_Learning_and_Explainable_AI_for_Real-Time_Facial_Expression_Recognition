# app.py
import os
import io
import time
import base64
import queue
import threading
import traceback

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import dlib

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import matplotlib.pyplot as plt

import sqlite3
import re

# ---------------- CONFIG ----------------
MODEL_PATH = "models/xception_emotion.pth"
DLIB_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
IMG_SIZE = 299
CLASSES = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
DEVICE = torch.device("cpu")

# Performance tuning (adjust for your machine)
GRADCAM_EVERY_N_FRAMES = 8
LIME_ENQUEUE_EVERY_N_FRAMES = 100
LIME_NUM_SAMPLES = 120
LIME_QUEUE_MAX = 2

RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- FLASK ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- MODEL & TRANSFORMS ----------------
print("Loading model...")
model = timm.create_model('xception', pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)
print("Model loaded.")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# dlib detector
detector = dlib.get_frontal_face_detector()
predictor = None
if os.path.exists(DLIB_PREDICTOR):
    predictor = dlib.shape_predictor(DLIB_PREDICTOR)

# ---------------- GradCAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx):
        # input_tensor: (1,C,H,W)
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward(retain_graph=False)
        grads = self.gradients.mean(dim=[2,3], keepdim=True)  # (B,C,1,1)
        cam = (grads * self.activations).sum(dim=1, keepdim=True)  # (B,1,Hf,Wf)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()  # (H,W)
        return cam

gradcam = GradCAM(model, model.conv4)

# ---------------- Helpers ----------------
def tensor_from_pil(pil_img):
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def predict_emotion_from_pil(pil_img):
    t = tensor_from_pil(pil_img)
    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return CLASSES[idx], idx, probs  # return full probs array

def safe_cam_to_uint8(cam):
    cam = np.array(cam)
    cam = np.squeeze(cam)
    if cam.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype='uint8')
    mn = np.nanmin(cam); mx = np.nanmax(cam)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-8:
        cam_norm = np.zeros_like(cam)
    else:
        cam_norm = (cam - mn) / (mx - mn)
    cam_u8 = (cam_norm * 255.0).astype('uint8')
    return cam_u8

def make_gradcam_overlay(pil_face, class_idx):
    inp = tensor_from_pil(pil_face)
    cam = gradcam.generate(inp, class_idx)
    cam_u8 = safe_cam_to_uint8(cam)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)  # BGR
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    face_np = np.array(pil_face).astype('uint8')
    overlay = np.uint8(0.6 * heatmap_rgb + 0.4 * face_np)
    return overlay

def render_prob_panel(probs, classes, width=IMG_SIZE, height=IMG_SIZE):
    """Render right-top panel with sorted top-5 probabilities."""
    bg = Image.new("RGB", (width, height), (8, 12, 18))
    draw = ImageDraw.Draw(bg)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), "Top Predictions", font=font, fill=(220,220,220))
    sorted_idx = np.argsort(probs)[::-1]
    y = 40
    for i in sorted_idx[:5]:
        lbl = classes[i]
        p = float(probs[i])
        bar_w = int((width - 140) * p)
        draw.rectangle([10, y, 10 + bar_w, y + 18], fill=(14, 165, 140))
        draw.rectangle([10 + bar_w, y, width - 10, y + 18], outline=(40,60,70))
        draw.text((width - 120, y), f"{p:.2f}", font=font, fill=(220,220,220))
        draw.text((12, y - 2), lbl, font=font, fill=(220,220,220))
        y += 28
    draw.text((10, height-22), time.strftime("%Y-%m-%d %H:%M:%S"), font=font, fill=(120,140,150))
    return np.array(bg)

# ---------------- LIME background worker ----------------
lime_explainer = lime_image.LimeImageExplainer()
lime_queue = queue.Queue(maxsize=LIME_QUEUE_MAX)
latest_lime_b64 = None
lime_lock = threading.Lock()

def batch_predict_for_lime(images):
    # images: list of HxWx3 uint8 arrays
    model.eval()
    batch = torch.stack([transform(Image.fromarray(i)).to(DEVICE) for i in images], dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

def lime_worker():
    global latest_lime_b64
    while True:
        try:
            face_np, ts = lime_queue.get()
        except Exception:
            time.sleep(0.1)
            continue
        try:
            if face_np is None or face_np.size == 0:
                continue
            face_uint8 = np.asarray(face_np).astype('uint8')
            # ensure bright enough / 3 channels
            if face_uint8.ndim == 2:
                face_uint8 = np.stack([face_uint8]*3, axis=-1)
            ex = lime_explainer.explain_instance(face_uint8, batch_predict_for_lime,
                                                top_labels=1, hide_color=0, num_samples=LIME_NUM_SAMPLES)
            temp, mask = ex.get_image_and_mask(ex.top_labels[0], positive_only=False, num_features=8, hide_rest=False)
            overlay = (mark_boundaries(temp/255.0, mask) * 255).astype(np.uint8)
            pil = Image.fromarray(overlay)
            b = io.BytesIO(); pil.save(b, format='PNG')
            b64 = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
            with lime_lock:
                latest_lime_b64 = b64
        except Exception as e:
            # fallback: edge-based overlay
            try:
                edges = cv2.Canny(cv2.cvtColor(face_uint8, cv2.COLOR_RGB2GRAY), 50, 150)
                edges3 = np.stack([edges]*3, axis=-1)
                pil = Image.fromarray(edges3)
                b = io.BytesIO(); pil.save(b, format='PNG')
                b64 = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
                with lime_lock:
                    latest_lime_b64 = b64
            except Exception:
                print("LIME worker fallback failed:", e)
        finally:
            try:
                lime_queue.task_done()
            except Exception:
                pass

threading.Thread(target=lime_worker, daemon=True).start()

# ---------------- Camera class ----------------
class VideoCamera:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.latest_jpeg = None
        self.frame_counter = 0
        self.last_gradcam = None
        self.last_lime_enqueue = 0
        self.last_probs = np.zeros(len(CLASSES), dtype=float)

    def start(self):
        with self.lock:
            if self.running:
                return
            # cv2.CAP_DSHOW helps on Windows; ignore on Linux
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if os.name == 'nt' else 0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam.")
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def _update_loop(self):
        global latest_lime_b64, lime_enabled
        last_time = time.time()
        while True:
            with self.lock:
                if not self.running:
                    break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05); continue
            self.frame_counter += 1

            processed = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            # default panels (top-left=video, top-right=probs panel, bottom-left=gradcam, bottom-right=lime)
            left_panel = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
            top_right = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            bottom_left = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            bottom_right = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            if len(faces) > 0:
                f = faces[0]
                x1, y1, x2, y2 = max(0,f.left()), max(0,f.top()), min(frame.shape[1], f.right()), min(frame.shape[0], f.bottom())
                pad = int(0.15 * max(x2-x1, y2-y1))
                x1e, y1e = max(0, x1-pad), max(0, y1-pad)
                x2e, y2e = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                face_crop = frame[y1e:y2e, x1e:x2e]
                if face_crop.size != 0:
                    face_rgb = cv2.cvtColor(cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
                    pil_face = Image.fromarray(face_rgb)

                    # predict
                    try:
                        label, idx, probs = predict_emotion_from_pil(pil_face)
                        self.last_probs = probs
                    except Exception as e:
                        label, idx, probs = ("err", 0, np.zeros(len(CLASSES)))
                        print("Predict realtime error:", e)

                    # draw bbox & top-1 label
                    cv2.rectangle(processed, (x1e, y1e), (x2e, y2e), (0,255,0), 2)
                    cv2.putText(processed, f"{label} {probs[idx]:.2f}", (x1e, max(20,y1e)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # GradCAM occasionally
                    if (self.frame_counter % GRADCAM_EVERY_N_FRAMES) == 0 or (self.last_gradcam is None):
                        try:
                            self.last_gradcam = make_gradcam_overlay(pil_face, idx)
                        except Exception as e:
                            print("GradCAM error:", e); self.last_gradcam = None
                    if self.last_gradcam is not None:
                        bottom_left = self.last_gradcam

                    # Enqueue LIME occasionally
                    if lime_enabled and (self.frame_counter - self.last_lime_enqueue) >= LIME_ENQUEUE_EVERY_N_FRAMES:
                        try:
                            if not lime_queue.full():
                                lime_queue.put_nowait((face_rgb.copy(), time.time()))
                                self.last_lime_enqueue = self.frame_counter
                        except Exception:
                            pass

                    # decode latest lime b64
                    with lime_lock:
                        b64 = latest_lime_b64
                    if b64:
                        try:
                            header, enc = b64.split(',',1)
                            arr = np.frombuffer(base64.b64decode(enc), dtype=np.uint8)
                            lime_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if lime_bgr is not None:
                                lime_rgb = cv2.cvtColor(lime_bgr, cv2.COLOR_BGR2RGB)
                                bottom_right = cv2.resize(lime_rgb, (IMG_SIZE, IMG_SIZE))
                        except Exception:
                            bottom_right = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

                    left_panel = face_rgb

            # render probability panel from last_probs
            top_right = render_prob_panel(self.last_probs, CLASSES, width=IMG_SIZE, height=IMG_SIZE)

            # combine 2x2 grid
            top_row = np.hstack([left_panel, top_right])
            bottom_row = np.hstack([bottom_left, bottom_right])
            combined = np.vstack([top_row, bottom_row])

            # overlay status and fps
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            last_time = now
            cv2.putText(combined, f"LIME={'ON' if lime_enabled else 'OFF'} | FPS: {fps:.1f}", (10, combined.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            with self.lock:
                self.latest_jpeg = jpeg.tobytes()

            time.sleep(0.02)

# ---------------- Flask routes ----------------
camera_obj = None
camera_lock = threading.Lock()
lime_enabled = True


@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    global camera_obj
    with camera_lock:
        if camera_obj is not None:
            return jsonify({"status":"already_running"})
        camera_obj = VideoCamera(0)
        try:
            camera_obj.start()
        except Exception as e:
            camera_obj = None
            return jsonify({"status":"error","message":str(e)}), 500
    return jsonify({"status":"started"})

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    global camera_obj
    with camera_lock:
        if camera_obj is None:
            return jsonify({"status":"not_running"})
        try:
            camera_obj.stop()
        except Exception as e:
            print("Error stopping camera:", e)
        camera_obj = None
    return jsonify({"status":"stopped"})

@app.route('/video_feed')
def video_feed():
    def gen():
        global camera_obj
        while True:
            with camera_lock:
                cam = camera_obj
            if cam is None:
                blank = np.zeros((IMG_SIZE*2, IMG_SIZE*2, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue
            frame = cam.get_jpeg()
            if frame is None:
                time.sleep(0.05); continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_lime', methods=['POST'])
def toggle_lime():
    global lime_enabled
    lime_enabled = not lime_enabled
    return jsonify({"lime_enabled": lime_enabled})

# ---------------- Static upload route ----------------
@app.route('/upload', methods=['GET','POST'])
def upload():
    global camera_obj
    if request.method == 'POST':
        with camera_lock:
            if camera_obj is not None:
                camera_obj.stop()
                camera_obj = None
        f = request.files.get('file')
        if not f:
            return redirect(url_for('index'))
        filename = f.filename
        save_path = os.path.join(RESULTS_DIR, filename)
        f.save(save_path)
        pil = Image.open(save_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        # Predict + GradCAM
        label, idx, probs = predict_emotion_from_pil(pil)
        grad_overlay = make_gradcam_overlay(pil, idx)

        # LIME (sync)
        try:
            face_np = np.array(pil).astype('uint8')
            ex = lime_explainer.explain_instance(face_np, batch_predict_for_lime,
                                                 top_labels=1, hide_color=0, num_samples=300)
            temp, mask = ex.get_image_and_mask(ex.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            lime_overlay = (mark_boundaries(temp/255.0, mask) * 255).astype(np.uint8)
            lime_pil = Image.fromarray(lime_overlay)
        except Exception as e:
            print("LIME static error:", e)
            lime_pil = None

        # SHAP (sync)
        try:
            def f_shap(x):
                arr = x.astype('float32')
                if arr.max() > 1.0:
                    arr = arr / 255.0
                mean = np.array([0.485,0.456,0.406]).reshape(1,1,3)
                std = np.array([0.229,0.224,0.225]).reshape(1,1,3)
                arr = (arr - mean) / std
                t = torch.from_numpy(arr.transpose(0,3,1,2)).float()
                with torch.no_grad():
                    out = model(t)
                    probs = F.softmax(out, dim=1).cpu().numpy()
                return probs

            expl = shap.Explainer(f_shap, np.zeros((1,IMG_SIZE,IMG_SIZE,3)))
            shap_vals = expl(np.expand_dims(np.array(pil), axis=0), max_evals=200)
            vals = shap_vals.values
            if isinstance(vals, list):
                arr = np.array(vals[idx])
            else:
                arr = vals[0]
            if arr.ndim == 3:
                agg = np.mean(arr, axis=2)
            else:
                agg = arr
            agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
            cmap = plt.get_cmap('jet')(agg)[:,:,:3]
            cmap_u8 = (cmap * 255).astype('uint8')
            shap_pil = Image.fromarray(cmap_u8)
        except Exception as e:
            print("SHAP error:", e)
            shap_pil = None

        # save outputs
        gpath = os.path.join(RESULTS_DIR, f"grad_{filename}")
        Image.fromarray(grad_overlay).save(gpath)
        lpath = None; spath = None
        if lime_pil:
            lpath = os.path.join(RESULTS_DIR, f"lime_{filename}")
            lime_pil.save(lpath)
        if shap_pil:
            spath = os.path.join(RESULTS_DIR, f"shap_{filename}")
            shap_pil.save(spath)

        return render_template('result.html', emotion=label, grad_path=gpath, lime_path=lpath, shap_path=spath)
    return render_template('home1.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    with camera_lock:
        if camera_obj is not None:
            camera_obj.stop()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Shutting down"


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/home1')
def home1():
    return render_template('home1.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('signin.html')


# ---------------- Run ----------------
if __name__ == "__main__":
    camera_lock = threading.Lock()
    app.run(debug=True, threaded=True)
