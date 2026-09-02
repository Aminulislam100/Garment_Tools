import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import ezdxf
from rembg import remove

# ==========================================
# ১. পেজ সেটআপ ও গ্লোবাল কনফিগারেশন
# ==========================================
st.set_page_config(page_title="Ultimate QC & Auto DXF Tool", layout="wide", page_icon="⚙️")

# ==========================================
# ২. প্রফেশনাল 3D CSS এবং থিম ফিক্স
# ==========================================
st.markdown("""
<style>
    /* 3D Top Navigation Bar Styling - Centered & Professional */
    div[data-testid="stRadio"] {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        background: #f1f5f9;
        padding: 8px;
        border-radius: 50px;
        box-shadow: inset 5px 5px 10px #cbd5e1, inset -5px -5px 10px #ffffff;
        gap: 15px;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] label {
        background: transparent;
        padding: 12px 40px;
        border-radius: 40px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    
    /* Hide Default Radio Circles */
    div[data-testid="stRadio"] > div[role="radiogroup"] label span[data-baseweb="radio"] {
        display: none;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] label p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        margin: 0;
    }
    
    /* Active State for 3D Navbar */
    div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(145deg, #1e40af, #3b82f6);
        box-shadow: 4px 4px 10px #93c5fd, -4px -4px 10px #ffffff;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"] p {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }

    /* Fixing Main Container */
    [data-testid="stContainer"] { 
        background-color: #ffffff !important; 
        color: #0f172a !important; 
    }
    
    /* Section Headers */
    .step-header-1 {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%);
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(194,65,12,0.3);
        margin-bottom: 15px;
    }
    
    .step-header-2 {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(4,120,87,0.3);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ৩. ডিপ লার্নিং ও গ্লোবাল ফাংশনস (Fabric AI & Image Optimizers)
# ==========================================
BENCHMARK_DIR = "benchmark"
os.makedirs(BENCHMARK_DIR, exist_ok=True)

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

# ==========================================
# 8. প্রফেশনাল ইমেজ ও ভেক্টর স্মুথিং ইঞ্জিন (For CAD/3D Pattern)
# ==========================================
def deep_enhance_and_highlight(pil_img):
    """
    ছবি অপ্টিমাইজেশন, ব্যাকগ্রাউন্ড রিমুভ, শার্পেনিং ও হাইলাইটেড ট্রেসিং ভিউ
    """
    # ১. OOM ফিক্স: নিরাপদ সাইজে রিসাইজ
    pil_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    
    # ২. কালার ভাইব্রেন্স ও কন্ট্রাস্ট এনহ্যান্সমেন্ট
    enhancer_col = ImageEnhance.Color(pil_img)
    pil_img = enhancer_col.enhance(1.25)
    enhancer_con = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer_con.enhance(1.35)
    
    # ৩. ব্যাকগ্রাউন্ড রিমুভ (rembg)
    img_no_bg = remove(pil_img)
    img_array = np.array(img_no_bg)
    
    # ৪. RGBA থেকে ক্লিয়ার RGB কনভার্সন
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        alpha_f = alpha[:, :, np.newaxis] / 255.0
        img_rgb = (rgb * alpha_f + white_bg * (1 - alpha_f)).astype(np.uint8)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.ndim == 3 else cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # ৫. বিল্যাটেরাল ফিল্টার (নয়েজ দূর করবে কিন্তু এজ/লাইন শার্প রাখবে)
    bgr_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr_img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # ৬. CLAHE অটোকন্ট্রাস্ট ও এডাপ্টিভ থ্রেশহোল্ডিং
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # ৭. মরফোলজিকাল ক্লোজিং (ভাঙ্গা লাইন জোড়া লাগাবে)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_gray = cv2.morphologyEx(enhanced_gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # ৮. এডাপ্টিভ বাইনারি থ্রেশহোল্ড
    thresh = cv2.adaptiveThreshold(
        closed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # ৯. কনটুর/আউটলাইন ডিটেকশন
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # ১০. হাইলাইটেড ট্রেসিং ওভারলে তৈরি (Electric Green Highlights)
    highlight_img = img_rgb.copy()
    cv2.drawContours(highlight_img, contours, -1, (0, 255, 128), 2)
    
    return img_rgb, highlight_img, contours

def export_smooth_dxf(contours, output_filename, smoothness_factor=0.002, min_area=30):
    """
    CAD এবং 3D Pattern Apps এর জন্য স্মুথ ও নিরবচ্ছিন্ন DXF লাইন জেনারেটর
    """
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    valid_count = 0

    for cnt in contours:
        # ১. ছোট নয়েজ বা ভাঙ্গা ডট বাদ দেওয়া
        area = cv2.contourArea(cnt)
        arc_len = cv2.arcLength(cnt, True)
        
        if area > min_area and arc_len > 15:
            # ২. Douglas-Peucker Smoothing (পিক্সেল ভাঙ্গা লাইন সোজা ও স্মুথ করবে)
            epsilon = smoothness_factor * arc_len
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx_cnt) >= 2:
                points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx_cnt]
                # DXF এ স্মুথ পলিলাইন যোগ
                msp.add_lwpolyline(points, close=True)
                valid_count += 1

    doc.saveas(output_filename)
    return valid_count

# ==========================================
# ৪. মেইন নেভিগেশন (Top Menu)
# ==========================================
app_mode = st.radio("Navigation", ["🧵 QC Checker", "📐 Auto DXF Converter"], horizontal=True, label_visibility="collapsed")

# ==========================================
# APP 1: QC CHECKER (Fabric Vision AI)
# ==========================================
if app_mode == "🧵 QC Checker":
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); padding: 32px 20px; border-radius: 18px; box-shadow: 0 15px 35px rgba(15, 23, 42, 0.4); border: 2px solid rgba(255, 255, 255, 0.15); text-align: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: 0.5px;">
            🧵 Advanced QC Checker
        </h1>
        <p style="color: #94a3b8; margin: 10px 0 0 0; font-size: 15px; font-weight: 600;">
            Deep Learning AI Vectors (Design) + CIELAB Spatial Grid (Shading)
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'captured_benchmarks' not in st.session_state:
        st.session_state.captured_benchmarks = []
    if 'last_cam_hash' not in st.session_state:
        st.session_state.last_cam_hash = None
    if 'cam_key' not in st.session_state:
        st.session_state.cam_key = 0

    st.sidebar.header("⚙️ Settings & Modes")
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
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ থ্রেশহোল্ড কাস্টমাইজেশন")
    st.sidebar.info("আপনার ফ্যাক্টরির মান অনুযায়ী আলাদা আলাদা পাস মার্ক সেট করুন।")
    
    color_threshold = st.sidebar.slider("🎨 কালার/শেড পাস মার্ক (%)", 50.0, 99.0, 78.0, 1.0)
    pattern_threshold = st.sidebar.slider("✨ প্রিন্ট/প্যাটার্ন পাস মার্ক (%)", 50.0, 99.0, 80.0, 1.0)
    texture_threshold = st.sidebar.slider("🧶 টেক্সচার/ঘনত্ব পাস মার্ক (%)", 50.0, 99.0, 70.0, 1.0)

    with st.container(border=True):
        st.markdown('<div class="step-header-1">🏆 ধাপ ১: বেঞ্চমার্ক ইনপুট (Master Sample Setup)</div>', unsafe_allow_html=True)
        
        bench_method = st.radio("মাস্টার স্যাম্পল কিভাবে দেবেন?", ("📁 গ্যালারি / ফাইল", "📸 লাইভ ক্যামেরা"), horizontal=True)
        st.markdown("---")
        
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
                st.info(f"📸 তোলা হয়েছে: {len(st.session_state.captured_benchmarks)} টি স্যাম্পল")
                col_save1, col_save2, col_save3 = st.columns(3)
                with col_save1:
                    if st.button("➕ বর্তমান স্যাম্পলের সাথে যোগ করুন", use_container_width=True):
                        existing_count = len([f for f in os.listdir(BENCHMARK_DIR)])
                        for i, img_bytes in enumerate(st.session_state.captured_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_cam_{existing_count + i + 1}.jpg"), "wb") as f: f.write(img_bytes)
                        st.session_state.captured_benchmarks = []; st.session_state.cam_key += 1
                        st.success("✅ যোগ করা হয়েছে!")
                        time.sleep(0.8)
                        st.rerun()
                with col_save2:
                    if st.button("🔄 আগের সব মুছে নতুন সেভ করুন", type="primary", use_container_width=True):
                        for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                        for i, img_bytes in enumerate(st.session_state.captured_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_cam_{i+1}.jpg"), "wb") as f: f.write(img_bytes)
                        st.session_state.captured_benchmarks = []; st.session_state.cam_key += 1
                        st.success("✅ নতুন সেভ হয়েছে!")
                        time.sleep(0.8)
                        st.rerun()
                with col_save3:
                    if st.button("❌ ক্যানসেল / রিটেক", use_container_width=True):
                        st.session_state.captured_benchmarks = []
                        st.session_state.last_cam_hash = None
                        st.session_state.cam_key += 1
                        st.rerun()

        st.markdown("---")
        updated_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if updated_files:
            st.write(f"**📂 সেভ করা মাস্টার স্যাম্পল ({len(updated_files)} টি):**")
            cols = st.columns(min(len(updated_files), 5) if len(updated_files) > 0 else 1)
            for idx, file in enumerate(updated_files):
                with cols[idx % 5]:
                    st.image(os.path.join(BENCHMARK_DIR, file), caption=file, use_container_width=True)
            if st.button("🗑️ সব মাস্টার স্যাম্পল মুছুন"):
                for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                st.rerun()

    st.write("") 
    with st.container(border=True):
        st.markdown('<div class="step-header-2">🔬 ধাপ ২: টেস্টিং স্ক্যানার (Production Check)</div>', unsafe_allow_html=True)
        final_benchmark_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not final_benchmark_files:
            st.error("⚠️ টেস্টিং শুরু করার আগে উপরে অন্তত ১টি মাস্টার স্যাম্পল সেভ করুন।")
        else:
            benchmark_data = load_cached_benchmarks(tuple(final_benchmark_files), BENCHMARK_DIR)
            vector_model = load_vector_model()

            col_test1, col_test2 = st.columns([1, 1.2])
            
            with col_test1:
                input_method = st.radio("কীভাবে স্ক্যান করবেন?", ("📁 গ্যালারি / ফাইল", "🔴 লাইভ ক্যামেরা"), horizontal=True)
                if input_method == "📁 গ্যালারি / ফাইল":
                    camera_image = st.file_uploader("টেস্টিংয়ের ছবি দিন", type=['png', 'jpg', 'jpeg'], key="test_upload", accept_multiple_files=False)
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
                        strict_pattern_threshold = 0.85 
                        if cosine_sim < strict_pattern_threshold:
                            pattern_pct = 0.0
                        else:
                            pattern_pct = ((cosine_sim - strict_pattern_threshold) / (1.0 - strict_pattern_threshold)) * 100.0
                        
                        lap_diff = abs(b_lap - cam_lap)
                        texture_pct = max(0.0, 100.0 - (lap_diff / (max(b_lap, 1e-5)) * 80.0))
                        
                        edge_diff = abs(b_edge - cam_edge)
                        edge_pct = max(0.0, 100.0 - (edge_diff / (max(b_edge, 1e-5)) * 100.0))
                        
                        final_texture_pct = (texture_pct * 0.6) + (edge_pct * 0.4)
                        
                        if "শুধুমাত্র কালার/শেডিং" in inspection_mode:
                            final_score = color_pct
                        elif "শুধুমাত্র ডিজাইন/প্রিনট" in inspection_mode:
                            final_score = pattern_pct
                        elif "সুতার ঘনত্ব / টেক্সচার" in inspection_mode:
                            final_score = final_texture_pct
                        elif "কালার + ডিজাইন" in inspection_mode:
                            final_score = (color_pct * 0.50) + (pattern_pct * 0.50)
                        else:  
                            final_score = (color_pct * 0.40) + (pattern_pct * 0.40) + (final_texture_pct * 0.20)
                        
                        if final_score > best_match_score:
                            best_match_score = final_score
                            best_match_path, best_match_name = b_path, name
                            d_color, d_pattern, d_texture = color_pct, pattern_pct, final_texture_pct
                    
                    st.write(f"**চেকিং মোড:** `{inspection_mode}`")
                    st.markdown(f"### **ওভারঅল একুরেসি:** `{best_match_score:.2f}%`")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("🎨 Color Score", f"{d_color:.1f}%", f"Target: {color_threshold}%")
                    col_m2.metric("✨ Pattern Score", f"{d_pattern:.1f}%", f"Target: {pattern_threshold}%")
                    col_m3.metric("🧶 Texture Score", f"{d_texture:.1f}%", f"Target: {texture_threshold}%")
                    
                    is_pass = False
                    fail_reasons = []

                    if "শুধুমাত্র কালার/শেডিং" in inspection_mode:
                        is_pass = d_color >= color_threshold
                        if not is_pass: fail_reasons.append("কালার/শেড ম্যাচ করেনি")
                    elif "শুধুমাত্র ডিজাইন/প্রিনট" in inspection_mode:
                        is_pass = d_pattern >= pattern_threshold
                        if not is_pass: fail_reasons.append("ডিজাইন/প্যাটার্ন ম্যাচ করেনি")
                    elif "সুতার ঘনত্ব / টেক্সচার" in inspection_mode:
                        is_pass = d_texture >= texture_threshold
                        if not is_pass: fail_reasons.append("টেক্সচার/ঘনত্ব ম্যাচ করেনি")
                    elif "কালার + ডিজাইন" in inspection_mode:
                        is_pass = (d_color >= color_threshold) and (d_pattern >= pattern_threshold)
                        if d_color < color_threshold: fail_reasons.append("কালার")
                        if d_pattern < pattern_threshold: fail_reasons.append("ডিজাইন")
                    else:  
                        is_pass = (d_color >= color_threshold) and (d_pattern >= pattern_threshold) and (d_texture >= texture_threshold)
                        if d_color < color_threshold: fail_reasons.append("কালার")
                        if d_pattern < pattern_threshold: fail_reasons.append("ডিজাইন")
                        if d_texture < texture_threshold: fail_reasons.append("টেক্সচার")

                    if is_pass:
                        st.success("### 🎉 PASS - প্রোডাক্ট কোয়ালিটি সঠিক আছে!")
                    else:
                        st.error(f"### ❌ FAIL - রিজেক্টেড! (ত্রুটি: {', '.join(fail_reasons)})")
                        
                    st.markdown("---")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), caption="আপনার স্ক্যান (Test)", use_container_width=True)
                    with v_col2:
                        if best_match_path:
                            st.image(cv2.cvtColor(cv2.imread(best_match_path), cv2.COLOR_BGR2RGB), caption=f"ম্যাচিং মাস্টার: {best_match_name}", use_container_width=True)

# ==========================================
# APP 2: AUTO DXF CONVERTER
# ==========================================
elif app_mode == "📐 Auto DXF Converter":
    
    st.sidebar.title("📐 DXF মেনু")
    page = st.sidebar.radio("আপনার প্রয়োজনীয় টুলটি বেছে নিন:", ["Convert Clear Photo", "Process Photo and Convert"])

    st.sidebar.markdown("---")
    st.sidebar.info("১. **Convert Clear Photo:** শুধু পরিষ্কার ছবির জন্য দ্রুত কনভার্টার।\n২. **Process Photo:** অস্পষ্ট ছবির ব্যাকগ্রাউন্ড রিমুভ, লাইন শার্পেনিং ও ৩D অ্যাপ অপ্টিমাইজড কনভার্টার।")

    if page == "Convert Clear Photo":
        st.title("📐 Convert Clear Photo")
        st.write("আপনার জ্যামিতিক বা টাইলসের পরিষ্কার ছবি আপলোড করুন অথবা লাইভ ক্যামেরা দিয়ে তুলুন, এক ক্লিকে DXF ডাউনলোড করুন।")

        tab1, tab2 = st.tabs(["📁 ফাইল আপলোড", "📸 লাইভ ক্যামেরা"])
        uploaded_file = None
        
        with tab1:
            up_file = st.file_uploader("একটি ছবি বাছাই করুন", type=["jpg", "jpeg", "png", "bmp"], key="dxf_up1")
            if up_file: uploaded_file = up_file
        with tab2:
            cam_file = st.camera_input("ক্যামেরা দিয়ে ছবি তুলুন", key="dxf_cam1")
            if cam_file: uploaded_file = cam_file

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    st.error("⚠️ ছবি সঠিকভাবে লোড করা সম্ভব হয়নি! আবার চেষ্টা করুন।")
                else:
                    st.image(img, caption="আপনার ইনপুট ছবি", width=350)

                    if st.button("⚡ DXF ফাইলে কনভার্ট করুন", key="btn1"):
                        with st.spinner("প্রসেসিং হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন..."):
                            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            doc = ezdxf.new(dxfversion="R2010")
                            msp = doc.modelspace()

                            count = 0
                            for cnt in contours:
                                arc_len = cv2.arcLength(cnt, True)
                                if arc_len > 15:
                                    # Smoothing to prevent broken vector lines
                                    approx = cv2.approxPolyDP(cnt, 0.002 * arc_len, True)
                                    points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                                    msp.add_lwpolyline(points, close=True)
                                    count += 1

                            if count == 0:
                                st.warning("⚠️ কোনো আউটলাইন/অবজেক্ট সনাক্ত করা যায়নি। অন্য পরিষ্কার ছবি চেষ্টা করুন।")
                            else:
                                file_base_name = getattr(uploaded_file, 'name', 'Camera_Snapshot.jpg')
                                output_filename = f"{os.path.splitext(file_base_name)[0]}.dxf"
                                doc.saveas(output_filename)

                                with open(output_filename, "rb") as file:
                                    st.success("সফলভাবে কনভার্ট হয়েছে!")
                                    st.download_button(
                                        label="📥 DXF ফাইল ডাউনলোড করুন",
                                        data=file,
                                        file_name=output_filename,
                                        mime="application/dxf",
                                        key="dl_btn1"
                                    )
            except Exception as e:
                st.error(f"❌ একটি টেকনিক্যাল সমস্যা হয়েছে: {e}")

    elif page == "Process Photo and Convert":
        st.title("⚙️ Process Photo and Convert (3D Pattern Optimized)")
        st.write("এই টুলটি স্বয়ংক্রিয়ভাবে ব্যাকগ্রাউন্ড রিমুভ, কালার ও অটো-কন্ট্রাস্ট ফিক্স, লাইন হাইলাইট এবং ৩D অ্যাপসের উপযোগী মসৃণ (Smooth) DXF তৈরি করে।")

        tab1, tab2 = st.tabs(["📁 ফাইল আপলোড", "📸 লাইভ ক্যামেরা"])
        uploaded_file = None
        
        with tab1:
            up_file = st.file_uploader("একটি ছবি বাছাই করুন", type=["jpg", "jpeg", "png", "bmp"], key="dxf_up2")
            if up_file: uploaded_file = up_file
        with tab2:
            cam_file = st.camera_input("ক্যামেরা দিয়ে ছবি তুলুন", key="dxf_cam2")
            if cam_file: uploaded_file = cam_file

        if uploaded_file is not None:
            # Session state reset if a new file is uploaded
            file_id = getattr(uploaded_file, 'name', 'camera') + str(getattr(uploaded_file, 'size', 0))
            if st.session_state.get("current_file_id") != file_id:
                st.session_state["current_file_id"] = file_id
                st.session_state["processed"] = False
                st.session_state["highlight_img"] = None
                st.session_state["contours"] = None

            try:
                uploaded_file.seek(0)
                input_image = Image.open(uploaded_file)
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.image(input_image, caption="অরিজিনাল ইনপুট ছবি", use_container_width=True)

                st.markdown("---")
                
                # Step 1 Button: Process & Highlight Lines
                if st.button("✨ ১. ছবি প্রসেস ও লাইন হাইলাইট করুন", key="btn_process", type="primary"):
                    with st.spinner("অ্যাডভান্সড প্রসেসিং চলছে (ব্যাকগ্রাউন্ড রিমুভ, শার্পেন ও লাইন ট্রেসিং)..."):
                        clean_rgb, highlight_img, contours = deep_enhance_and_highlight(input_image)
                        st.session_state["highlight_img"] = highlight_img
                        st.session_state["contours"] = contours
                        st.session_state["processed"] = True

                # Step 2 View & DXF Generation
                if st.session_state.get("processed", False) and st.session_state["highlight_img"] is not None:
                    with col_p2:
                        st.image(
                            st.session_state["highlight_img"], 
                            caption="প্রসেসড ট্রেসিং ভিউ (হাইলাইটেড গ্রিন লাইন)", 
                            use_container_width=True
                        )

                    st.markdown("---")
                    st.success("✅ ছবি প্রসেসিং সম্পন্ন হয়েছে! নিচের বাটনে ক্লিক করে স্মুথ DXF ডাউনলোড করুন।")
                    
                    # Fine Tuning Slider for 3D Pattern Smoothness
                    smoothness = st.slider("🎛️ লাইন স্মুথনেস (Smoothness Level)", 0.001, 0.010, 0.0025, 0.0005, help="মান বাড়ালে ভাঙ্গা ভাঙ্গা বা করাত-দাঁতি লাইন সোজা ও স্মুথ হবে।")
                    
                    if st.button("📐 ২. স্মুথ DXF ফাইলে কনভার্ট করুন", key="btn_convert"):
                        with st.spinner("৩D অ্যাপের উপযোগী স্মুথ DXF তৈরি হচ্ছে..."):
                            file_base_name = getattr(uploaded_file, 'name', 'Camera_Snapshot.jpg')
                            output_filename = f"{os.path.splitext(file_base_name)[0]}_Smooth_3D.dxf"
                            
                            valid_lines = export_smooth_dxf(
                                st.session_state["contours"], 
                                output_filename, 
                                smoothness_factor=smoothness
                            )

                            if valid_lines == 0:
                                st.warning("⚠️ কোনো আউটলাইন পাওয়া যায়নি। ছবি পরিবর্তন করে আবার চেষ্টা করুন।")
                            else:
                                with open(output_filename, "rb") as file:
                                    st.balloons()
                                    st.success(f"🎉 সফলভাবে {valid_lines} টি স্মুথ ভেক্টর কার্ভের নিখুঁত DXF তৈরি হয়েছে!")
                                    st.download_button(
                                        label="📥 স্মুথ 3D-রেডি DXF ডাউনলোড করুন",
                                        data=file,
                                        file_name=output_filename,
                                        mime="application/dxf",
                                        key="dl_btn2"
                                    )

            except Exception as e:
                st.error(f"❌ প্রসেস করার সময় একটি সমস্যা হয়েছে: {e}")
