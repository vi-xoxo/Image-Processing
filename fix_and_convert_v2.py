"""
fix_and_convert_v2.py
=====================
Dibuat khusus berdasarkan arsitektur notebook Untitled21.ipynb.

Model asli MENGANDUNG:
  - data_augmentation layer (RandomFlip, RandomRotation, RandomZoom)
  - resnet_v2.preprocess_input di dalam graph
  - ResNet50V2 base
  - GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> Dropout -> Dense(7)

Cara pakai:
    pip install tf2onnx
    python fix_and_convert_v2.py

Output: emotion_detector_final.onnx
"""

import h5py, numpy as np, os, sys

H5_PATH   = "emotion_detector_final.h5"
ONNX_PATH = "emotion_detector_final.onnx"
NUM_CLASSES = 7
IMG_SHAPE   = (224, 224, 3)

if not os.path.exists(H5_PATH):
    print(f"❌ File tidak ditemukan: {H5_PATH}")
    sys.exit(1)

print(f"📂 File: {H5_PATH} ({os.path.getsize(H5_PATH)/1024/1024:.1f} MB)")

# ─── 1. Cetak SELURUH struktur H5 ────────────────────────────────────────────
print("\n─── STEP 1: Full H5 Structure ───")
with h5py.File(H5_PATH, "r") as f:
    print("Root attrs:", list(f.attrs.keys()))
    print("Root keys:", list(f.keys()))

    def print_tree(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[D] {name.split('/')[-1]}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{indent}[G] {name.split('/')[-1]}/")
    
    for key in f.keys():
        f[key].visititems(print_tree)
        print(f"  Root group: {key}/")

# ─── 2. Import TensorFlow ─────────────────────────────────────────────────────
print("\n─── STEP 2: Import TensorFlow ───")
import tensorflow as tf
print(f"  TensorFlow: {tf.__version__}")

# ─── 3. Rebuild arsitektur PERSIS seperti notebook ───────────────────────────
print("\n─── STEP 3: Rebuild Model (arsitektur persis notebook) ───")

# Data augmentation HARUS ada karena notebook menyertakannya dalam model
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="sequential")

base_model = tf.keras.applications.ResNet50V2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights=None,
    name="resnet50v2"
)
base_model.trainable = True
# Fine-tune: hanya 20 layer terakhir yang trainable (seperti notebook)
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.applications.resnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

print(f"  Model layers: {len(model.layers)}")
print(f"  Model params: {model.count_params():,}")

# ─── 4. Coba load weights ─────────────────────────────────────────────────────
print("\n─── STEP 4: Load Weights ───")

# Metode A: load_model langsung
try:
    model_loaded = tf.keras.models.load_model(H5_PATH, compile=False)
    model = model_loaded
    print("  ✅ load_model berhasil!")
except Exception as e:
    print(f"  ⚠️ load_model gagal: {e}")
    
    # Metode B: load_weights langsung (positional)
    try:
        model.load_weights(H5_PATH)
        print("  ✅ load_weights positional berhasil!")
    except Exception as e2:
        print(f"  ⚠️ load_weights positional gagal: {e2}")
        
        # Metode C: baca manual dari model_weights
        print("  🔄 Mencoba baca manual dari model_weights...")
        loaded = 0
        with h5py.File(H5_PATH, "r") as f:
            if "model_weights" not in f:
                print("  ❌ Tidak ada 'model_weights'")
            else:
                mw = f["model_weights"]
                print(f"  Keys di model_weights: {list(mw.keys())[:10]}")
                
                layer_map = {l.name: l for l in model.layers}
                for lname in mw.keys():
                    if lname not in layer_map:
                        continue
                    lg = mw[lname]
                    
                    # Kumpulkan semua dataset secara rekursif
                    arrays = []
                    def collect(name, obj):
                        if isinstance(obj, h5py.Dataset) and obj.shape != ():
                            arrays.append(np.array(obj))
                    lg.visititems(collect)
                    
                    if not arrays:
                        continue
                    
                    layer = layer_map[lname]
                    if len(arrays) == len(layer.weights):
                        try:
                            layer.set_weights(arrays)
                            loaded += 1
                            print(f"    ✅ {lname}: {len(arrays)} weights")
                        except Exception as ex:
                            print(f"    ❌ {lname}: {ex}")
                    else:
                        print(f"    ⚠️ {lname}: count mismatch ({len(arrays)} vs {len(layer.weights)})")
        
        if loaded == 0:
            print("\n  ❌ Semua metode loading gagal.")
            print("  📋 Paste seluruh output ini ke chat untuk diagnosa lebih lanjut.")
            sys.exit(1)
        else:
            print(f"\n  ✅ {loaded} layer berhasil dimuat secara manual.")

# ─── 5. Test prediksi ─────────────────────────────────────────────────────────
print("\n─── STEP 5: Test Prediksi ───")
dummy = np.zeros((1, *IMG_SHAPE), dtype=np.float32)
pred = model(dummy, training=False).numpy()
print(f"  Output shape: {pred.shape}, sum: {pred.sum():.4f}")

# ─── 6. Konversi ke ONNX ─────────────────────────────────────────────────────
print("\n─── STEP 6: Konversi ke ONNX ───")
try:
    import tf2onnx

    # Coba from_keras dulu (lebih reliable untuk model dengan custom layers)
    try:
        tf2onnx.convert.from_keras(model, output_path=ONNX_PATH, opset=13)
    except Exception as e_keras:
        print(f"  ⚠️ from_keras gagal ({e_keras}), mencoba from_function...")
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")
        ])
        def serving(x):
            return model(x, training=False)
        serving.get_concrete_function()
        tf2onnx.convert.from_function(
            serving,
            input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input")],
            output_path=ONNX_PATH,
            opset=13,
        )
    print(f"  ✅ ONNX disimpan: {ONNX_PATH}")

    # Verifikasi
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    print(f"  ✅ Verifikasi OK. Output: {out.shape}, sum={out.sum():.4f}")
    print(f"\n🎉 SELESAI! Jalankan: streamlit run app.py")

except ImportError:
    print("  ❌ tf2onnx belum terinstall.")
    print("  Jalankan: pip install tf2onnx")
except Exception as e:
    print(f"  ❌ Konversi gagal: {e}")
