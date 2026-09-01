import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ==========================================
# গ্লোবাল পেজ কনফিগারেশন (এটি সবার আগে থাকতে হবে)
# ==========================================
st.set_page_config(page_title="Smart AI Suite", layout="wide", page_icon="⚙️")

# ==========================================
# সাইডবার নেভিগেশন তৈরি
# ==========================================
st.sidebar.title("🎛️ Main Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "কাজের ধরন নির্বাচন করুন:",
    ("🧵 Fabric Vision AI", "📐 DXF Converter")
)
st.sidebar.markdown("---")

# ==========================================
# পেজ ১: Fabric Vision AI এর সম্পূর্ণ কোড
# ==========================================
def fabric_vision_page():
    # কাস্টম স্টাইলিং
    st.markdown("""
    <style>
    [data-testid="stContainer"] { border-radius: 0 0 16px 16px !important; padding: 24px !important; background-color: #ffffff !important; margin-bottom: 30px !important; }
    [data-testid="stContainer"]:nth-of-type(1) { border: 6px solid #c2410c !important; border-top: none !important; box-shadow: 0 15px 35px rgba(194, 65, 12, 0.22) !important; }
    [data-testid="stContainer"]:nth-of-type(2) { border: 6px solid #047857 !important; border-top: none !important; box-shadow: 0 15px 35px rgba(4, 120, 87, 0.22) !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); padding: 32px 20px; border-radius: 18px; box-shadow: 0 15px 35px rgba(15, 23, 42, 0.4); border: 2px solid rgba(255, 255, 255, 0.15); text-align: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: 0.5px;">
            🧵 Advanced Fabric AI Hybrid Checker
        </h1>
        <p style="color: #94a3b8; margin: 10px 0 0 0; font-size: 15px; font-weight: 600;">
            Deep Learning AI Vectors (Design) + CIELAB Spatial Grid (Shading)
        </p>
    </div>
    """, unsafe_allow_html=True)

    BENCHMARK_DIR = "benchmark"
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    if 'captured_benchmarks' not in st.session_state:
        st.session_state.captured_benchmarks = []
    if 'last_cam_hash' not in st.session_state:
        st.session_state.last_cam_hash = None
    if 'cam_key' not in st.session_state:
        st.session_state.cam_key = 0

    @st.cache_resource(show_spinner="AI ভেক্টর ইঞ্জিন লোড হচ্ছে...")
    def load_vector_model():
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        return model

    def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def extract_hybrid_features(cv_bgr_img, vector_model):
        h, w = cv_bgr_img.shape[:2]
        h_step, w_step = max(1, h // 3), max(1, w // 3)
        hist_features = []
        for i in range(3):
            for j in range(3):
                sub_crop = cv_bgr_img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                if sub_crop.size > 0:
                    blur = cv2.GaussianBlur(sub_crop, (7, 7), 0)
                    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
                    hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(hist, hist)
                    hist_features.extend(hist.flatten())
        lab_hist = np.array(hist_features, dtype=np.float32)

        pil_img = Image.fromarray(cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor_img = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            embedding = vector_model(tensor_img).squeeze().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        gray = cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = (np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])) * 100.0

        return lab_hist, embedding, laplacian_var, edge_density

    @st.cache_data(show_spinner=False)
    def load_cached_benchmarks(file_list, bench_dir):
        vector_model = load_vector_model()
        data = []
        for b_file in file_list:
            b_path = os.path.join(bench_dir, b_file)
            img = cv2.imread(b_path)
            if img is not None:
                img = resize_with_aspect_ratio(img, width=500)
                lab_hist, embedding, lap_var, edge_density = extract_hybrid_features(img, vector_model)
                data.append((b_file, b_path, lab_hist, embedding, lap_var, edge_density))
        return data

    st.sidebar.header("⚙️ Fabric Settings")
    inspection_mode = st.sidebar.selectbox(
        "🔍 ইনস্পেকশন মোড বেছে নিন",
        (
            "🌟 হাইব্রিড অল-ইন-ওয়ান (শেড + ভেক্টর ডিজাইন + টেক্সচার)", 
            "🎨 কালার + ডিজাইন (LAB + AI Vector)", 
            "👕 শুধুমাত্র কালার/শেডিং (CIELAB Grid)", 
            "✨ শুধুমাত্র ডিজাইন/প্রিন্ট (AI Vector Embedding)",
            "🧶 সুতার ঘনত্ব / টেক্সচার (Density Check)"
        )
    )
    pass_threshold = st.sidebar.slider("Minimum Match Score (%)", 50.0, 99.0, 78.0, 1.0)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%); padding: 16px 22px; border-radius: 14px 14px 0 0; color: white; font-weight: bold; font-size: 18px; box-shadow: 0 6px 15px rgba(194,65,12,0.35); border: 6px solid #c2410c; border-bottom: none;">
        🏆 ধাপ ১: বেঞ্চমার্ক ইনপুট (Master Sample Setup)
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        bench_method = st.radio("মাস্টার স্যাম্পল কিভাবে দেবেন?", ("📁 গ্যালারি / ফাইল", "📸 লাইভ ক্যামেরা"), horizontal=True)
        if bench_method == "📁 গ্যালারি / ফাইল":
            new_benchmarks = st.file_uploader("ছবি আপলোড করুন", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="bench_upload")
            if new_benchmarks:
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("➕ বর্তমান স্যাম্পলের সাথে যোগ করুন", use_container_width=True):
                        existing_count = len([f for f in os.listdir(BENCHMARK_DIR)])
                        for i, file in enumerate(new_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_{existing_count + i + 1}.jpg"), "wb") as f:
                                f.write(file.getbuffer())
                        st.success("✅ যোগ করা হয়েছে!")
                        time.sleep(0.8)
                        st.rerun()
                with col_btn2:
                    if st.button("🔄 আগের সব মুছে নতুন সেভ করুন", type="primary", use_container_width=True):
                        for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                        for i, file in enumerate(new_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_{i+1}.jpg"), "wb") as f:
                                f.write(file.getbuffer())
                        st.success("✅ নতুন স্যাম্পল সেভ হয়েছে!")
                        time.sleep(0.8)
                        st.rerun()
                with col_btn3:
                    if st.button("❌ ক্যানসেল / বাদ দিন", use_container_width=True):
                        st.rerun()
        else:
            bench_cam = st.camera_input("মাস্টার স্যাম্পলের ছবি তুলুন", key=f"bench_cam_input_{st.session_state.cam_key}")
            if bench_cam:
                cam_bytes = bench_cam.getvalue()
                if st.session_state.last_cam_hash != cam_bytes:
                    st.session_state.captured_benchmarks.append(cam_bytes)
                    st.session_state.last_cam_hash = cam_bytes
            if len(st.session_state.captured_benchmarks) > 0:
                col_save1, col_save2, col_save3 = st.columns(3)
                with col_save1:
                    if st.button("➕ যোগ করুন", use_container_width=True):
                        existing_count = len([f for f in os.listdir(BENCHMARK_DIR)])
                        for i, img_bytes in enumerate(st.session_state.captured_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_cam_{existing_count + i + 1}.jpg"), "wb") as f: f.write(img_bytes)
                        st.session_state.captured_benchmarks = []; st.session_state.cam_key += 1
                        st.rerun()
                with col_save2:
                    if st.button("🔄 নতুন সেভ করুন", type="primary", use_container_width=True):
                        for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                        for i, img_bytes in enumerate(st.session_state.captured_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_cam_{i+1}.jpg"), "wb") as f: f.write(img_bytes)
                        st.session_state.captured_benchmarks = []; st.session_state.cam_key += 1
                        st.rerun()
                with col_save3:
                    if st.button("❌ ক্যানসেল", use_container_width=True):
                        st.session_state.captured_benchmarks = []; st.session_state.last_cam_hash = None; st.session_state.cam_key += 1
                        st.rerun()

        updated_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if updated_files:
            st.write(f"**📂 সেভ করা মাস্টার স্যাম্পল ({len(updated_files)} টি):**")
            if st.button("🗑️ সব মুছুন"):
                for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                st.rerun()

    st.markdown("""
    <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 16px 22px; border-radius: 14px 14px 0 0; color: white; font-weight: bold; font-size: 18px; box-shadow: 0 6px 15px rgba(4,120,87,0.35); border: 6px solid #047857; border-bottom: none;">
        🔬 ধাপ ২: টেস্টিং স্ক্যানার (Production Check)
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        if not updated_files:
            st.error("⚠️ টেস্টিং শুরু করার আগে উপরে অন্তত ১টি মাস্টার স্যাম্পল সেভ করুন।")
        else:
            benchmark_data = load_cached_benchmarks(tuple(updated_files), BENCHMARK_DIR)
            vector_model = load_vector_model()
            col_test1, col_test2 = st.columns([1, 1.2])
            with col_test1:
                input_method = st.radio("কীভাবে স্ক্যান করবেন?", ("📁 গ্যালারি / ফাইল", "🔴 লাইভ ক্যামেরা"), horizontal=True)
                if input_method == "📁 গ্যালারি / ফাইল":
                    camera_image = st.file_uploader("টেস্টিংয়ের ছবি দিন", type=['png', 'jpg', 'jpeg'], key="test_upload")
                else:
                    camera_image = st.camera_input("লাইভ স্ক্যান করুন", key="test_cam")
            with col_test2:
                if camera_image:
                    bytes_data = camera_image.getvalue()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    cv_img = resize_with_aspect_ratio(cv_img, width=500)
                    cam_hist, cam_embed, cam_lap, cam_edge = extract_hybrid_features(cv_img, vector_model)
                    
                    best_match_score = 0.0
                    best_match_path, best_match_name = "", ""
                    d_color, d_pattern, d_texture = 0.0, 0.0, 0.0
                    
                    for name, b_path, b_hist, b_embed, b_lap, b_edge in benchmark_data:
                        b_distance = cv2.compareHist(b_hist, cam_hist, cv2.HISTCMP_BHATTACHARYYA)
                        color_pct = max(0.0, (1.0 - (b_distance * 1.5)) * 100.0)
                        
                        cosine_sim = np.dot(b_embed, cam_embed)
                        pattern_pct = 0.0 if cosine_sim < 0.75 else ((cosine_sim - 0.75) / 0.25) * 100.0
                        
                        lap_diff = abs(b_lap - cam_lap)
                        texture_pct = max(0.0, 100.0 - (lap_diff / (max(b_lap, 1e-5)) * 150.0))
                        
                        if "শুধুমাত্র কালার" in inspection_mode: final_score = color_pct
                        elif "শুধুমাত্র ডিজাইন" in inspection_mode: final_score = pattern_pct
                        elif "সুতার ঘনত্ব" in inspection_mode: final_score = texture_pct
                        elif "কালার + ডিজাইন" in inspection_mode: final_score = (color_pct * 0.5) + (pattern_pct * 0.5)
                        else: final_score = (color_pct * 0.4) + (pattern_pct * 0.4) + (texture_pct * 0.2)
                        
                        if final_score > best_match_score:
                            best_match_score = final_score
                            best_match_path, best_match_name = b_path, name
                            d_color, d_pattern, d_texture = color_pct, pattern_pct, texture_pct
                    
                    st.write(f"**চেকিং মোড:** `{inspection_mode}`")
                    st.markdown(f"### **ফাইনাল একুরেসি:** `{best_match_score:.2f}%`")
                    if best_match_score >= pass_threshold:
                        st.success("### 🎉 PASS - প্রোডাক্ট কোয়ালিটি সঠিক আছে!")
                    else:
                        st.error("### ❌ FAIL - রিজেক্টেড!")

# ==========================================
# পেজ ২: DXF Converter এর প্লেসহোল্ডার
# ==========================================
def dxf_converter_page():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 32px 20px; border-radius: 18px; text-align: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0;">📐 DXF Image Converter</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("আপনার DXF_Converter এর app.py ফাইলের ভেতরের কোডটি আমাকে দেননি। আপনি এই ফাংশনের ভেতরে আপনার DXF এর কোডটি বসিয়ে নিতে পারেন, অথবা আমাকে কোডটি দিলে আমি এখানে যুক্ত করে দেব।")
    # TODO: Paste the DXF conversion logic here

# ==========================================
# অ্যাপ রাউটিং (কোন পেজটি দেখাবে)
# ==========================================
if app_mode == "🧵 Fabric Vision AI":
    fabric_vision_page()
elif app_mode == "📐 DXF Converter":
    dxf_converter_page()
