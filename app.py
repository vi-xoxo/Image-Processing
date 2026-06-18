import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense — Emotion Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 1.8rem 1rem 1.2rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(59,130,246,0.15));
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 24px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Tab Pills ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(167,139,250,0.2);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 50px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Upload Area ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8) !important;
    background: rgba(124,58,237,0.08) !important;
}
[data-testid="stFileUploader"] label {
    color: #c4b5fd !important;
    font-weight: 600 !important;
}

/* ── Camera ── */
[data-testid="stCameraInput"] {
    border-radius: 20px;
    overflow: hidden;
}
[data-testid="stCameraInput"] video {
    border-radius: 16px !important;
    border: 2px solid rgba(167,139,250,0.3) !important;
}
[data-testid="stCameraInputButton"] {
    background: linear-gradient(135deg, #f472b6, #a78bfa, #60a5fa) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 64px !important;
    height: 64px !important;
    box-shadow: 0 0 20px rgba(167,139,250,0.6) !important;
    transition: transform 0.2s ease !important;
}
[data-testid="stCameraInput"] label { display: none; }

/* ── Result Image ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ── Emotion Card (compact) ── */
.emo-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.2));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 14px;
    padding: 0.65rem 1rem;
    text-align: center;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.emo-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa, #34d399);
}
.emo-emoji { font-size: 1.8rem; margin-bottom: 0.1rem; }
.emo-label {
    font-size: 1.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
}
.emo-conf {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.15rem;
    font-weight: 400;
}
.emo-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 0.1rem 0.6rem;
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── Divider ── */
hr { border-color: rgba(167,139,250,0.15) !important; margin: 1.5rem 0 !important; }

/* ── Alert boxes ── */
.stSuccess { background: rgba(52,211,153,0.1) !important; border: 1px solid rgba(52,211,153,0.3) !important; border-radius: 12px !important; }
.stWarning { background: rgba(251,191,36,0.1) !important; border: 1px solid rgba(251,191,36,0.3) !important; border-radius: 12px !important; }
.stError   { background: rgba(248,113,113,0.1) !important; border: 1px solid rgba(248,113,113,0.3) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Config ──────────────────────────────────────────────────────────────
@st.cache_data
def load_class_mapping():
    with open("class_mapping.json", "r") as f:
        return json.load(f)

class_mapping = load_class_mapping()
NUM_CLASSES   = len(class_mapping)

EMOTION_EMOJI = {
    "anger": "😠", "contempt": "😒", "disgust": "🤢",
    "fear": "😨", "happy": "😊", "sadness": "😢", "surprise": "😲"
}
# Gradient warna per emosi (BGR untuk OpenCV)
EMOTION_COLORS_BGR = {
    "anger":    (50,  50,  240),
    "contempt": (180, 60,  200),
    "disgust":  (30,  180, 80),
    "fear":     (30,  140, 240),
    "happy":    (30,  220, 120),
    "sadness":  (220, 100, 30),
    "surprise": (220, 200, 30),
}
# Warna chart (hex)
EMOTION_COLORS_HEX = {
    "anger":    "#f87171",
    "contempt": "#c084fc",
    "disgust":  "#4ade80",
    "fear":     "#38bdf8",
    "happy":    "#fbbf24",
    "sadness":  "#fb923c",
    "surprise": "#f472b6",
}

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_yolo():
    import os
    if os.path.exists("yolov8n-face.pt"):
        try:
            return YOLO("yolov8n-face.pt"), None
        except:
            pass
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        p = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection",
                            filename="model.pt", local_dir=".")
        shutil.copy(p, "yolov8n-face.pt")
        return YOLO("yolov8n-face.pt"), None
    except:
        return None, YOLO("yolov8n.pt")

@st.cache_resource
def load_emotion_model():
    import os
    if os.path.exists("emotion_detector_final.onnx"):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession("emotion_detector_final.onnx",
                                        providers=["CPUExecutionProvider"])
            return {"type": "onnx", "session": sess}
        except Exception as e:
            st.warning(f"⚠️ ONNX gagal: {e}")
    if not os.path.exists("emotion_detector_final.h5"):
        st.error("❌ File model tidak ditemukan.")
        return None
    try:
        import tensorflow as tf
        data_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name="sequential")
        base = tf.keras.applications.ResNet50V2(
            input_shape=(224,224,3), include_top=False, weights=None, name="resnet50v2")
        inp = tf.keras.Input(shape=(224,224,3))
        x   = data_aug(inp)
        x   = tf.keras.applications.resnet_v2.preprocess_input(x)
        x   = base(x, training=False)
        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        x   = tf.keras.layers.Dense(256, activation='relu')(x)
        x   = tf.keras.layers.BatchNormalization()(x)
        x   = tf.keras.layers.Dropout(0.4)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inp, out)
        model.load_weights("emotion_detector_final.h5")
        return {"type": "keras_tf", "model": model}
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None

face_yolo, fallback_yolo = load_face_yolo()
emotion_bundle = load_emotion_model()

# ─── Prediksi dengan TTA ──────────────────────────────────────────────────────
def enhance_face(img_bgr):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization):
    Meningkatkan detail lokal wajah — sangat membantu untuk
    emosi subtle (sadness, anger, disgust, fear) yang bergantung
    pada detail otot wajah yang kecil.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l_eq  = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def predict_emotion(face_bgr):
    if emotion_bundle is None:
        return np.ones(NUM_CLASSES) / NUM_CLASSES

    def infer(img):
        rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(rgb.astype("float32"), axis=0)
        if emotion_bundle["type"] == "onnx":
            sess = emotion_bundle["session"]
            return sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
        return emotion_bundle["model"](arr, training=False).numpy()[0]

    def softmax(x, T=1.0):
        e = np.exp((x - np.max(x)) / T)
        return e / e.sum()

    h, w = face_bgr.shape[:2]
    enhanced = enhance_face(face_bgr)

    # CLAHE lebih agresif untuk emosi subtle
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
    enhanced2 = cv2.merge([clahe2.apply(l), a, b])
    enhanced2 = cv2.cvtColor(enhanced2, cv2.COLOR_LAB2BGR)

    preds  = []
    wts    = []

    # 1. Original — bobot tinggi
    preds.append(infer(face_bgr));   wts.append(1.5)

    # 2. CLAHE standar — kunci sadness/anger/fear
    preds.append(infer(enhanced));   wts.append(1.6)

    # 3. CLAHE agresif — memperjelas kerutan & lipatan mikro
    preds.append(infer(enhanced2));  wts.append(1.4)

    # 4. Flip horizontal + CLAHE
    preds.append(infer(cv2.flip(enhanced, 1)));  wts.append(1.0)

    # 5. Brightness lebih terang
    preds.append(infer(cv2.convertScaleAbs(face_bgr, alpha=1.25, beta=15)));  wts.append(0.8)

    # 6. Contrast tinggi + sharpen — lipatan ekspresi lebih jelas
    sharp_k = np.array([[0,-0.5,0],[-0.5,3.0,-0.5],[0,-0.5,0]])
    sharpened_enh = cv2.filter2D(enhanced2, -1, sharp_k)
    preds.append(infer(sharpened_enh));  wts.append(1.2)

    # 7. Crop mata+alis+dahi (45% atas) — sangat penting fear/anger/sadness
    #    Alis turun = anger, alis naik+mata lebar = fear, sudut dalam naik = sadness
    eye_region = face_bgr[0:int(h*0.50), :]
    if eye_region.size > 0:
        preds.append(infer(eye_region));  wts.append(1.3)
        # Versi CLAHE dari region mata
        eye_enh = enhanced2[0:int(h*0.50), :]
        if eye_enh.size > 0:
            preds.append(infer(eye_enh));  wts.append(1.2)

    # 8. Crop area mulut+hidung (45% bawah) — disgust/contempt/sadness
    mouth_region = face_bgr[int(h*0.45):, :]
    if mouth_region.size > 0:
        preds.append(infer(mouth_region));  wts.append(0.9)

    # 9. Crop pipi kiri & kanan (cek asimetri wajah — penting anger/contempt)
    cheek_l = face_bgr[int(h*0.25):int(h*0.75), 0:int(w*0.5)]
    cheek_r = face_bgr[int(h*0.25):int(h*0.75), int(w*0.5):]
    if cheek_l.size > 0:
        preds.append(infer(cheek_l));  wts.append(0.6)
    if cheek_r.size > 0:
        preds.append(infer(cheek_r));  wts.append(0.6)

    # 10. Center crop 90%
    m = int(min(h,w)*0.05)
    center = face_bgr[m:h-m, m:w-m]
    if center.size > 0:
        preds.append(infer(center));  wts.append(0.8)

    # 11. Gamma correction gelap (untuk foto over-exposed)
    gamma = 1.4
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype="uint8")
    dark_gamma = cv2.LUT(enhanced, lut)
    preds.append(infer(dark_gamma));  wts.append(0.7)

    wts_arr = np.array(wts[:len(preds)]) / sum(wts[:len(preds)])
    avg_pred = np.average(preds, axis=0, weights=wts_arr)

    # ── Temperature scaling: kurangi dominansi "happy" yang berlebihan ──────────
    # Terapkan temperature > 1 untuk membuat distribusi lebih rata,
    # lalu re-boost emosi negatif (sadness, anger, fear) secara proporsional
    EMOTION_LIST = [class_mapping[str(k)] for k in range(len(avg_pred))]
    SAD_IDX   = EMOTION_LIST.index("sadness")   if "sadness"   in EMOTION_LIST else -1
    ANG_IDX   = EMOTION_LIST.index("anger")     if "anger"     in EMOTION_LIST else -1
    FEAR_IDX  = EMOTION_LIST.index("fear")      if "fear"      in EMOTION_LIST else -1
    HAPPY_IDX = EMOTION_LIST.index("happy")     if "happy"     in EMOTION_LIST else -1

    # Boost ringan untuk emosi negatif jika skor mereka cukup signifikan (>8%)
    # Ini mengimbangi bias dataset terhadap "happy"
    boosted = avg_pred.copy()
    for idx, thr, factor in [
        (SAD_IDX,  0.08, 1.35),
        (ANG_IDX,  0.08, 1.30),
        (FEAR_IDX, 0.07, 1.30),
    ]:
        if idx >= 0 and boosted[idx] > thr:
            boosted[idx] *= factor
    # Normalisasi ulang agar tetap valid probabilitas
    boosted = np.clip(boosted, 0, None)
    boosted = boosted / boosted.sum()

    return boosted

# ─── Bounding Box Gaya Modern ─────────────────────────────────────────────────
def draw_modern_box(img, x1, y1, x2, y2, emotion, conf):
    color = EMOTION_COLORS_BGR.get(emotion, (167,139,250))
    emoji = EMOTION_EMOJI.get(emotion, "")
    label = f"{emotion.upper()}  {conf:.1f}%"
    w_box = x2 - x1
    h_box = y2 - y1
    cl    = min(int(w_box * 0.18), int(h_box * 0.18), 28)
    thick = 3

    # Glow effect — gambar kotak sedikit lebih tebal dengan opacity rendah
    overlay = img.copy()
    cv2.rectangle(overlay, (x1-1, y1-1), (x2+1, y2+1), color, 2)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Corner brackets dengan anti-aliasing
    for (px, py), (ex, ey) in [
        ((x1, y1+cl),(x1,y1)),  ((x1,y1),(x1+cl,y1)),       # TL
        ((x2-cl,y1),(x2,y1)),   ((x2,y1),(x2,y1+cl)),        # TR
        ((x1,y2-cl),(x1,y2)),   ((x1,y2),(x1+cl,y2)),        # BL
        ((x2-cl,y2),(x2,y2)),   ((x2,y2),(x2,y2-cl)),        # BR
    ]:
        cv2.line(img, (px,py), (ex,ey), color, thick, cv2.LINE_AA)

    # Label pill background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
    pad  = 9
    lx1  = x1
    ly1  = max(y1 - th - pad*2 - 2, 0)
    lx2  = x1 + tw + pad*2
    ly2  = max(y1 - 2, th + pad*2)

    # Background pill semi-transparan dengan gradient feel
    pill = img.copy()
    cv2.rectangle(pill, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(pill, 0.72, img, 0.28, 0, img)

    # Teks label
    cv2.putText(img, label, (lx1+pad, ly2-pad),
                font, font_scale, (255,255,255), 2, cv2.LINE_AA)
    return img

# ─── Deteksi ──────────────────────────────────────────────────────────────────
def detect_and_predict(image_np):
    try:
        out   = image_np.copy()
        results = []
        model_used = face_yolo if face_yolo is not None else fallback_yolo
        is_face_model = face_yolo is not None

        # Sharpening ringan untuk foto blur/kamera
        kernel = np.array([[0,-0.3,0],[-0.3,2.2,-0.3],[0,-0.3,0]])
        sharpened = cv2.filter2D(image_np, -1, kernel)

        if is_face_model:
            yolo_res = model_used(sharpened, conf=0.35, verbose=False)
        else:
            yolo_res = model_used(sharpened, conf=0.35, classes=[0], verbose=False)

        boxes = yolo_res[0].boxes

        if boxes is None or len(boxes) == 0:
            preds   = predict_emotion(image_np)
            top_idx = int(np.argmax(preds))
            results.append({"emotion": class_mapping[str(top_idx)],
                            "confidence": preds, "box": None})
        else:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pw = int((x2-x1)*0.12)
                ph = int((y2-y1)*0.12)

                if is_face_model:
                    # YOLOv8-face: box sudah di wajah, tambah padding saja
                    fx1 = max(0, x1-pw)
                    fy1 = max(0, y1-ph)
                    fx2 = min(image_np.shape[1], x2+pw)
                    fy2 = min(image_np.shape[0], y2+ph)
                else:
                    # YOLO person: ambil 55% atas + padding
                    fx1 = max(0, x1-pw)
                    fy1 = max(0, y1)
                    fx2 = min(image_np.shape[1], x2+pw)
                    fy2 = min(image_np.shape[0], y1+int((y2-y1)*0.55))

                face_crop = image_np[fy1:fy2, fx1:fx2]
                if face_crop.size == 0:
                    continue

                preds   = predict_emotion(face_crop)
                top_idx = int(np.argmax(preds))
                emotion = class_mapping[str(top_idx)]
                conf    = preds[top_idx] * 100

                results.append({"emotion": emotion, "confidence": preds,
                                "box": (x1, y1, x2, y2)})
                out = draw_modern_box(out, x1, y1, x2, y2, emotion, conf)

        return out, results
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return image_np, []

# ─── Tampilkan Hasil ──────────────────────────────────────────────────────────
def show_results(results):
    for idx, r in enumerate(results):
        emotion  = r["emotion"]
        preds    = r["confidence"]
        emoji    = EMOTION_EMOJI.get(emotion, "🎭")
        top_conf = float(preds[np.argmax(preds)]) * 100

        if len(results) > 1:
            st.markdown(f"<p style='color:#94a3b8;font-weight:600;margin:0'>👤 Orang #{idx+1}</p>",
                        unsafe_allow_html=True)

        st.markdown(f"""
        <div class="emo-card">
            <div class="emo-emoji">{emoji}</div>
            <div class="emo-label">{emotion.upper()}</div>
            <div class="emo-conf">Keyakinan <b style="color:#e2e8f0">{top_conf:.1f}%</b></div>
            <span class="emo-badge">ResNet50V2 + TTA</span>
        </div>
        """, unsafe_allow_html=True)

        # Chart horizontal gradient
        labels = [class_mapping[str(k)] for k in range(len(preds))]
        values = [float(p)*100 for p in preds]
        colors = [EMOTION_COLORS_HEX.get(labels[j], "#a78bfa") for j in range(len(labels))]

        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)

        # Highlight bar emosi aktif dengan glow
        for bar, lbl in zip(bars, labels):
            if lbl == emotion:
                bar.set_linewidth(2)
                bar.set_edgecolor("white")
                bar.set_alpha(1.0)
            else:
                bar.set_alpha(0.45)

        ax.set_xlim(0, 115)
        ax.set_xlabel("Keyakinan (%)", color="#64748b", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#64748b")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_tick_params(color="#1e293b")
        ax.yaxis.set_tick_params(length=0)
        ax.grid(axis='x', color=(1,1,1,0.06), linewidth=0.5, zorder=0)

        for bar, val in zip(bars, values):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center",
                    color="white" if val > 5 else "#64748b",
                    fontsize=8, fontweight="600")

        plt.tight_layout(pad=0.5)
        st.pyplot(fig, transparent=True)
        plt.close()

        if idx < len(results)-1:
            st.markdown("<hr>", unsafe_allow_html=True)

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🎭 EmoSense</div>
    <div class="hero-sub">Deteksi ekspresi wajah real-time · YOLOv8 + ResNet50V2</div>
</div>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
for k in ["last_id","last_results","last_output",
          "multi_ids","multi_results","multi_outputs","multi_labels"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─── Input Tabs ───────────────────────────────────────────────────────────────
tab_cam, tab_upload = st.tabs(["📷  Kamera", "📁  Upload Gambar"])

img_bgr      = None
source_label = ""
image_id     = None

with tab_cam:
    st.markdown("<p style='color:#94a3b8;font-size:0.85rem;margin-bottom:0.5rem'>"
                "📌 Posisikan wajah di tengah frame, pastikan pencahayaan cukup</p>",
                unsafe_allow_html=True)
    camera_img = st.camera_input("", key="camera", label_visibility="collapsed")
    if camera_img is not None:
        image_id     = f"cam_{hash(camera_img.getvalue())}"
        pil_img      = Image.open(camera_img).convert("RGB")
        img_bgr      = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        source_label = "📷 Foto dari Kamera"

with tab_upload:
    st.markdown("<p style='color:#94a3b8;font-size:0.85rem;margin-bottom:0.5rem'>"
                "📌 Upload hingga <b style=\"color:#a78bfa\">4 gambar</b> sekaligus — hasil ditampilkan terpisah</p>",
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["jpg","jpeg","png"],
                                      key="uploader", label_visibility="collapsed",
                                      accept_multiple_files=True)
    # Batasi maks 4
    if uploaded_files and len(uploaded_files) > 4:
        st.warning("⚠️ Maksimal 4 gambar. Hanya 4 gambar pertama yang diproses.")
        uploaded_files = uploaded_files[:4]

# ─── Mode tunggal (kamera atau 1 file) ───────────────────────────────────────
single_mode = False
if img_bgr is not None:
    single_mode = True
elif uploaded_files and len(uploaded_files) == 1:
    single_mode  = True
    uf           = uploaded_files[0]
    image_id     = f"up_{uf.name}_{uf.size}"
    pil_img      = Image.open(uf).convert("RGB")
    img_bgr      = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    source_label = f"📁 {uf.name}"

# ─── Mode Multi (2–4 gambar) ─────────────────────────────────────────────────
multi_mode = uploaded_files and len(uploaded_files) >= 2

# ════════════════════════════════════════════════════════════════════════════
#  SINGLE MODE
# ════════════════════════════════════════════════════════════════════════════
if single_mode and img_bgr is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    if image_id != st.session_state.last_id:
        with st.spinner("🔍 Menganalisis ekspresi wajah..."):
            out_img, res = detect_and_predict(img_bgr)
        st.session_state.last_id     = image_id
        st.session_state.last_results = res
        st.session_state.last_output  = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        # reset multi
        st.session_state.multi_ids = None

    st.markdown("<p style='color:#94a3b8;font-weight:600;font-size:0.9rem;"
                "margin-bottom:0.5rem'>🔍 Hasil Deteksi</p>", unsafe_allow_html=True)
    st.image(st.session_state.last_output, caption=source_label,
             use_container_width=True)

    if st.session_state.last_results:
        show_results(st.session_state.last_results)
    else:
        st.warning("⚠️ Tidak ada wajah yang terdeteksi. Coba foto dengan wajah lebih jelas.")

# ════════════════════════════════════════════════════════════════════════════
#  MULTI MODE
# ════════════════════════════════════════════════════════════════════════════
elif multi_mode:
    st.markdown("<hr>", unsafe_allow_html=True)
    n = len(uploaded_files)

    # Buat composite ID untuk cek apakah perlu re-proses
    combo_id = "_".join(f"{uf.name}_{uf.size}" for uf in uploaded_files)

    if combo_id != st.session_state.multi_ids:
        multi_outputs = []
        multi_results = []
        multi_labels  = []
        with st.spinner(f"🔍 Menganalisis {n} gambar..."):
            for uf in uploaded_files:
                pil  = Image.open(uf).convert("RGB")
                bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                out, res = detect_and_predict(bgr)
                multi_outputs.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                multi_results.append(res)
                multi_labels.append(uf.name)
        st.session_state.multi_ids     = combo_id
        st.session_state.multi_outputs = multi_outputs
        st.session_state.multi_results = multi_results
        st.session_state.multi_labels  = multi_labels
        # reset single
        st.session_state.last_id = None

    # ── Tampilkan grid gambar ─────────────────────────────────────────────
    st.markdown("<p style='color:#a78bfa;font-weight:700;font-size:1rem;"
                "margin-bottom:0.8rem'>🖼️ Hasil Deteksi — {} Gambar</p>".format(n),
                unsafe_allow_html=True)

    imgs   = st.session_state.multi_outputs
    labels = st.session_state.multi_labels
    res_all = st.session_state.multi_results

    # Tampilkan gambar dalam grid 2 kolom
    for row_start in range(0, n, 2):
        cols = st.columns(2)
        for col_i, abs_i in enumerate(range(row_start, min(row_start+2, n))):
            with cols[col_i]:
                st.markdown(
                    f"<p style='color:#94a3b8;font-weight:600;font-size:0.78rem;"
                    f"margin-bottom:0.2rem;text-align:center'>Hasil {abs_i+1} · {labels[abs_i]}</p>",
                    unsafe_allow_html=True)
                st.image(imgs[abs_i], use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tampilkan barchart 4 kolom berdampingan ───────────────────────────
    st.markdown(
        "<p style='color:#a78bfa;font-weight:700;font-size:1rem;"
        "margin-bottom:0.8rem'>📊 Perbandingan Emosi</p>",
        unsafe_allow_html=True)

    chart_cols = st.columns(n)
    for ci, res_list in enumerate(res_all):
        with chart_cols[ci]:
            st.markdown(
                f"<p style='color:#94a3b8;font-weight:600;font-size:0.78rem;"
                f"text-align:center;margin-bottom:0.2rem'>Hasil {ci+1}</p>",
                unsafe_allow_html=True)
            if not res_list:
                st.markdown(
                    "<p style='color:#f87171;font-size:0.75rem;text-align:center'>"
                    "⚠️ Tidak ada wajah</p>", unsafe_allow_html=True)
                continue

            # Ambil hasil pertama (wajah dominan)
            r        = res_list[0]
            emotion  = r["emotion"]
            preds    = r["confidence"]
            emoji    = EMOTION_EMOJI.get(emotion, "🎭")
            top_conf = float(preds[np.argmax(preds)]) * 100

            # Card kecil
            st.markdown(f"""
            <div class="emo-card">
                <div class="emo-emoji">{emoji}</div>
                <div class="emo-label">{emotion.upper()}</div>
                <div class="emo-conf"><b style="color:#e2e8f0">{top_conf:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Barchart vertikal (lebih ringkas untuk multi-mode)
            bar_labels = [class_mapping[str(k)] for k in range(len(preds))]
            bar_values = [float(p)*100 for p in preds]
            colors     = [EMOTION_COLORS_HEX.get(bar_labels[j], "#a78bfa")
                          for j in range(len(bar_labels))]

            fig, ax = plt.subplots(figsize=(2.8, 3.2))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            bars = ax.bar(range(len(bar_labels)), bar_values,
                          color=colors, width=0.6, zorder=3)
            for bar, lbl in zip(bars, bar_labels):
                bar.set_alpha(1.0 if lbl == emotion else 0.42)
                if lbl == emotion:
                    bar.set_edgecolor("white")
                    bar.set_linewidth(1.5)

            ax.set_xticks(range(len(bar_labels)))
            ax.set_xticklabels([l[:3].upper() for l in bar_labels],
                               fontsize=6.5, color="#94a3b8", rotation=45, ha="right")
            ax.set_ylim(0, 110)
            ax.tick_params(colors="#94a3b8", labelsize=6.5, left=False)
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis='y', color=(1,1,1,0.06), linewidth=0.5, zorder=0)

            for bar, val in zip(bars, bar_values):
                if val > 4:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                            f"{val:.0f}%", ha="center", va="bottom",
                            color="white", fontsize=5.5, fontweight="600")

            plt.tight_layout(pad=0.3)
            st.pyplot(fig, transparent=True)
            plt.close()

# ════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER (belum ada input)
# ════════════════════════════════════════════════════════════════════════════
else:
    # Reset session saat tidak ada input
    st.session_state.last_id      = None
    st.session_state.last_results = None
    st.session_state.last_output  = None
    st.session_state.multi_ids    = None

    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;opacity:0.4">
        <div style="font-size:4rem">🎭</div>
        <p style="color:#94a3b8;margin-top:0.5rem">
            Ambil foto atau upload 1–4 gambar untuk memulai deteksi emosi
        </p>
    </div>
    """, unsafe_allow_html=True)
