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

# ==========================================
# Crash-Proof Import for Rembg
# ==========================================
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    st.warning("⚠️ Rembg ইনস্টল করা নেই! ব্যাকগ্রাউন্ড রিমুভ ছাড়া প্রসেস হবে। ইনস্টল করতে: pip install rembg")

# ==========================================
# ১. পেজ সেটআপ ও গ্লোবাল কনফিগারেশন
# ==========================================
st.set_page_config(page_title="Ultimate QC & Auto DXF Tool", layout="wide", page_icon="⚙️")

st.markdown("""
<style>
    div[data-testid="stRadio"] { display: flex; justify-content: center; align-items: center; margin-bottom: 2rem; }
    div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex; flex-direction: row; justify-content: center; background: #f1f5f9; padding: 8px; border-radius: 50px; box-shadow: inset 5px 5px 10px #cbd5e1, inset -5px -5px 10px #ffffff; gap: 15px; }
    div[data-testid="stRadio"] > div[role="radiogroup"] label { background: transparent; padding: 12px 40px; border-radius: 40px; cursor: pointer; transition: all 0.3s ease; border: none; }
    div[data-testid="stRadio"] > div[role="radiogroup"] label span[data-baseweb="radio"] { display: none; }
    div[data-testid="stRadio"] > div[role="radiogroup"] label p { color: #475569 !important; font-weight: 700 !important; font-size: 20px !important; margin: 0; }
    div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"] { background: linear-gradient(145deg, #1e40af, #3b82f6); box-shadow: 4px 4px 10px #93c5fd, -4px -4px 10px #ffffff; }
    div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"] p { color: #ffffff !important; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); }
    [data-testid="stContainer"] { background-color: #ffffff !important; color: #0f172a !important; }
    .step-header-1 { background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%); padding: 12px 20px; border-radius: 8px; color: white; font-weight: bold; font-size: 18px; margin-bottom: 15px; }
    .step-header-2 { background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 12px 20px; border-radius: 8px; color: white; font-weight: bold; font-size: 18px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ২. ডিপ লার্নিং ও গ্লোবাল ফাংশনস (QC Checker)
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
    if width is None and height is None: return image
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
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tensor_img = preprocess(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = vector_model(tensor_img).squeeze().numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0: embedding = embedding / norm

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
# ৩. Pixlr-Style ফিল্টার ও প্রসেসিং ইঞ্জিন
# ==========================================
def process_base_image(pil_img):
    """ব্যাকগ্রাউন্ড রিমুভ ইঞ্জিন"""
    try:
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
        pil_img.thumbnail((1024, 1024), resample_filter)
        if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
        
        enhancer_col = ImageEnhance.Color(pil_img)
        pil_img = enhancer_col.enhance(1.1)
        
        img_array = None
        if REMBG_AVAILABLE:
            try:
                img_no_bg = remove(pil_img)
                img_array = np.array(img_no_bg)
            except Exception as e:
                img_array = np.array(pil_img)
        else:
            img_array = np.array(pil_img)
        
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            alpha = img_array[:, :, 3] / 255.0
            rgb = img_array[:, :, :3]
            white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
            img_rgb = (rgb * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        else:
            img_rgb = img_array

        return img_rgb, "Success"
    except Exception as e:
        return None, str(e)

def apply_pixlr_enhancements(img_rgb, do_auto_contrast=True, sharpen_pct=100, do_autofix=True):
    """Pixlr অ্যাপের মতো Auto Contrast, Sharpen 100%, এবং AutoFix প্রসেসিং"""
    processed = img_rgb.copy()
    
    # ১. AutoFix (Dynamic Range Normalization)
    if do_autofix:
        channels = cv2.split(processed)
        out_channels = []
        for ch in channels:
            min_val, max_val = np.percentile(ch, (1, 99))
            if max_val > min_val:
                ch_norm = np.clip((ch - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
            else:
                ch_norm = ch
            out_channels.append(ch_norm)
        processed = cv2.merge(out_channels)

    # ২. Auto Contrast (LAB Space CLAHE)
    if do_auto_contrast:
        lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # ৩. Sharpen (Unsharp Masking for Sharp Fine Details and Edges)
    if sharpen_pct > 0:
        factor = sharpen_pct / 100.0
        blurred = cv2.GaussianBlur(processed, (0, 0), 3)
        sharpened = cv2.addWeighted(processed, 1.0 + factor * 1.5, blurred, -factor * 1.5, 0)
        processed = np.clip(sharpened, 0, 255).astype(np.uint8)

    return processed

def extract_and_draw_lines_live(img_rgb, edge_sensitivity, smoothness, min_line_length):
    """হাইব্রিড এজ ডিটেকশন (Adaptive Thresholding + Canny)"""
    bgr_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    sens_norm = edge_sensitivity / 100.0

    # Bilateral filter used to keep edge boundaries sharp
    filtered_gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive Thresholding for subtle pattern features & outlines
    c_val = max(1, int(12 - (sens_norm * 10)))
    adaptive_edges = cv2.adaptiveThreshold(
        filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, c_val
    )

    # Dynamic Canny Edge Detection
    low_canny = int(max(5, 120 - (sens_norm * 110)))
    high_canny = int(max(25, 220 - (sens_norm * 160)))
    canny_edges = cv2.Canny(filtered_gray, low_canny, high_canny)

    # Combine both edge features
    combined_edges = cv2.bitwise_or(adaptive_edges, canny_edges)

    # Morphological clean up
    kernel = np.ones((2, 2), np.uint8)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(combined_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    highlight_img = img_rgb.copy()
    processed_contours = []

    for cnt in contours:
        arc_len = cv2.arcLength(cnt, False)
        if arc_len >= min_line_length:
            epsilon = smoothness * arc_len
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, False)
            processed_contours.append(approx_cnt)
            # Draw detected edges in green
            cv2.drawContours(highlight_img, [approx_cnt], -1, (0, 255, 0), 1)

    return highlight_img, processed_contours

def export_smooth_dxf(contours, output_filename):
    """DXF এ এক্সপোর্ট করার ফাংশন (SPLINE সমর্থনসহ)"""
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    valid_count = 0

    for cnt in contours:
        points = [(float(pt[0][0]), float(-pt[0][1]), 0) for pt in cnt]
        
        if len(points) >= 4:
            msp.add_spline(points)
            valid_count += 1
        elif len(points) >= 2:
            pts_2d = [(p[0], p[1]) for p in points]
            msp.add_lwpolyline(pts_2d, close=False)
            valid_count += 1
                
    doc.saveas(output_filename)
    return valid_count

# ==========================================
# ৪. মেইন নেভিগেশন (Top Menu)
# ==========================================
app_mode = st.radio("Navigation", ["🧵 QC Checker", "📐 Auto DXF Converter"], horizontal=True, label_visibility="collapsed")

# ==========================================
# APP 1: QC CHECKER
# ==========================================
if app_mode == "🧵 QC Checker":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); padding: 32px 20px; border-radius: 18px; box-shadow: 0 15px 35px rgba(15, 23, 42, 0.4); text-align: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 800;">🧵 Advanced QC Checker</h1>
    </div>
    """, unsafe_allow_html=True)

    if 'captured_benchmarks' not in st.session_state: st.session_state.captured_benchmarks = []
    if 'last_cam_hash' not in st.session_state: st.session_state.last_cam_hash = None
    if 'cam_key' not in st.session_state: st.session_state.cam_key = 0

    st.sidebar.header("⚙️ Settings")
    inspection_mode = st.sidebar.selectbox("🔍 ইনস্পেকশন মোড", ("🌟 হাইব্রিড অল-ইন-ওয়ান", "🎨 কালার + ডিজাইন", "👕 শুধুমাত্র কালার", "✨ শুধুমাত্র ডিজাইন", "🧶 সুতার ঘনত্ব"))
    color_threshold = st.sidebar.slider("🎨 কালার/শেড পাস মার্ক (%)", 50.0, 99.0, 78.0, 1.0)
    pattern_threshold = st.sidebar.slider("✨ প্রিন্ট/প্যাটার্ন পাস মার্ক (%)", 50.0, 99.0, 80.0, 1.0)
    texture_threshold = st.sidebar.slider("🧶 টেক্সচার/ঘনত্ব পাস মার্ক (%)", 50.0, 99.0, 70.0, 1.0)

    with st.container(border=True):
        st.markdown('<div class="step-header-1">🏆 ধাপ ১: বেঞ্চমার্ক ইনপুট</div>', unsafe_allow_html=True)
        bench_method = st.radio("মাস্টার স্যাম্পল কিভাবে দেবেন?", ("📁 গ্যালারি / ফাইল", "📸 লাইভ ক্যামেরা"), horizontal=True)
        
        if bench_method == "📁 গ্যালারি / ফাইল":
            new_benchmarks = st.file_uploader("ছবি আপলোড করুন", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="bench_upload")
            if new_benchmarks:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("➕ বর্তমান স্যাম্পলের সাথে যোগ করুন", use_container_width=True):
                        existing_count = len([f for f in os.listdir(BENCHMARK_DIR)])
                        for i, file in enumerate(new_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_{existing_count + i + 1}.jpg"), "wb") as f: f.write(file.getbuffer())
                        st.rerun()
                with col_btn2:
                    if st.button("🔄 আগের সব মুছে নতুন সেভ করুন", type="primary", use_container_width=True):
                        for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                        for i, file in enumerate(new_benchmarks):
                            with open(os.path.join(BENCHMARK_DIR, f"master_{i+1}.jpg"), "wb") as f: f.write(file.getbuffer())
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
                if st.button("🔄 সেভ করুন", type="primary"):
                    for f in os.listdir(BENCHMARK_DIR): os.remove(os.path.join(BENCHMARK_DIR, f))
                    for i, img_bytes in enumerate(st.session_state.captured_benchmarks):
                        with open(os.path.join(BENCHMARK_DIR, f"master_cam_{i+1}.jpg"), "wb") as f: f.write(img_bytes)
                    st.session_state.captured_benchmarks = []
                    st.session_state.cam_key += 1
                    st.rerun()

        updated_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if updated_files:
            st.write(f"**📂 সেভ করা স্যাম্পল ({len(updated_files)} টি):**")
            cols = st.columns(min(len(updated_files), 5) if len(updated_files) > 0 else 1)
            for idx, file in enumerate(updated_files):
                with cols[idx % 5]: st.image(os.path.join(BENCHMARK_DIR, file), caption=file, use_container_width=True)

    st.write("") 
    with st.container(border=True):
        st.markdown('<div class="step-header-2">🔬 ধাপ ২: টেস্টিং স্ক্যানার</div>', unsafe_allow_html=True)
        final_benchmark_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not final_benchmark_files:
            st.error("⚠️ টেস্টিং শুরু করার আগে ১টি মাস্টার স্যাম্পল সেভ করুন।")
        else:
            benchmark_data = load_cached_benchmarks(tuple(final_benchmark_files), BENCHMARK_DIR)
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
                    best_match_path = ""
                    d_color, d_pattern, d_texture = 0.0, 0.0, 0.0
                    
                    for name, b_path, b_hist, b_embed, b_lap, b_edge in benchmark_data:
                        b_distance = cv2.compareHist(b_hist, cam_hist, cv2.HISTCMP_BHATTACHARYYA)
                        color_pct = max(0.0, (1.0 - (b_distance * 1.5)) * 100.0)
                        
                        cosine_sim = np.dot(b_embed, cam_embed)
                        pattern_pct = 0.0 if cosine_sim < 0.85 else ((cosine_sim - 0.85) / 0.15) * 100.0
                        
                        texture_pct = max(0.0, 100.0 - (abs(b_lap - cam_lap) / (max(b_lap, 1e-5)) * 80.0))
                        edge_pct = max(0.0, 100.0 - (abs(b_edge - cam_edge) / (max(b_edge, 1e-5)) * 100.0))
                        final_texture_pct = (texture_pct * 0.6) + (edge_pct * 0.4)
                        
                        final_score = (color_pct * 0.40) + (pattern_pct * 0.40) + (final_texture_pct * 0.20)
                        
                        if final_score > best_match_score:
                            best_match_score, best_match_path = final_score, b_path
                            d_color, d_pattern, d_texture = color_pct, pattern_pct, final_texture_pct
                    
                    st.markdown(f"### **ওভারঅল একুরেসি:** `{best_match_score:.2f}%`")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("🎨 Color", f"{d_color:.1f}%")
                    col_m2.metric("✨ Pattern", f"{d_pattern:.1f}%")
                    col_m3.metric("🧶 Texture", f"{d_texture:.1f}%")

# ==========================================
# APP 2: AUTO DXF CONVERTER
# ==========================================
elif app_mode == "📐 Auto DXF Converter":
    st.sidebar.title("📐 DXF মেনু")
    page = st.sidebar.radio("আপনার প্রয়োজনীয় টুলটি বেছে নিন:", ["Convert Clear Photo", "Process Photo and Convert"])

    if page == "Convert Clear Photo":
        st.title("📐 Convert Clear Photo")
        tab1, tab2 = st.tabs(["📁 ফাইল আপলোড", "📸 লাইভ ক্যামেরা"])
        uploaded_file = None
        
        with tab1:
            up_file = st.file_uploader("ছবি বাছাই করুন", type=["jpg", "jpeg", "png", "bmp"], key="dxf_up1")
            if up_file: uploaded_file = up_file
        with tab2:
            cam_file = st.camera_input("ক্যামেরা দিয়ে ছবি তুলুন", key="dxf_cam1")
            if cam_file: uploaded_file = cam_file

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            st.image(img, caption="আপনার ইনপুট ছবি", width=350)

            if st.button("⚡ DXF ফাইলে কনভার্ট করুন"):
                with st.spinner("প্রসেসিং হচ্ছে..."):
                    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = cnts[0] if len(cnts) == 2 else cnts[1]

                    doc = ezdxf.new(dxfversion="R2010")
                    msp = doc.modelspace()
                    
                    for cnt in contours:
                        arc_len = cv2.arcLength(cnt, True)
                        if arc_len > 15:
                            approx = cv2.approxPolyDP(cnt, 0.002 * arc_len, True)
                            points = [(float(pt[0][0]), float(-pt[0][1]), 0) for pt in approx]
                            
                            if len(points) >= 4:
                                points.append(points[0])
                                msp.add_spline(points)
                            elif len(points) >= 2:
                                pts_2d = [(p[0], p[1]) for p in points]
                                msp.add_lwpolyline(pts_2d, close=True)

                    file_base_name = getattr(uploaded_file, 'name', 'Snapshot.jpg')
                    output_filename = f"{os.path.splitext(file_base_name)[0]}.dxf"
                    doc.saveas(output_filename)

                    with open(output_filename, "rb") as file:
                        st.download_button("📥 DXF ফাইল ডাউনলোড করুন", data=file, file_name=output_filename, mime="application/dxf")

    elif page == "Process Photo and Convert":
        st.title("⚙️ Process Photo and Convert (Pixlr Enhanced Edge Detection)")
        tab1, tab2 = st.tabs(["📁 ফাইল আপলোড", "📸 লাইভ ক্যামেরা"])
        uploaded_file = None
        
        with tab1:
            up_file = st.file_uploader("ছবি বাছাই করুন", type=["jpg", "jpeg", "png"], key="dxf_up2")
            if up_file: uploaded_file = up_file
        with tab2:
            cam_file = st.camera_input("ক্যামেরা দিয়ে ছবি তুলুন", key="dxf_cam2")
            if cam_file: uploaded_file = cam_file

        if uploaded_file is not None:
            file_id = getattr(uploaded_file, 'name', 'camera') + str(getattr(uploaded_file, 'size', 0))
            if st.session_state.get("current_file_id") != file_id:
                st.session_state["current_file_id"] = file_id
                st.session_state["clean_rgb"] = None

            uploaded_file.seek(0)
            input_image = Image.open(uploaded_file)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.image(input_image, caption="অরিজিনাল ইনপুট ছবি", use_container_width=True)

            st.markdown("---")
            
            # Step 1: Background removal
            if st.session_state.get("clean_rgb") is None:
                if st.button("✨ ১. ব্যাকগ্রাউন্ড রিমুভ ও ক্লিন করুন", type="primary"):
                    with st.spinner("অ্যাডভান্সড প্রসেসিং চলছে..."):
                        clean_rgb, status = process_base_image(input_image)
                        if clean_rgb is not None:
                            st.session_state["clean_rgb"] = clean_rgb
                            st.rerun()
                        else:
                            st.error(f"❌ প্রসেসিং ক্র্যাশ করেছে! কারণ: {status}")
            else:
                st.success("✅ ছবি প্রসেস হয়ে গেছে! এবার নিচের Pixlr অপশন ও স্লাইডার দিয়ে ডিজাইন বা অবজেক্টের আউটলাইন পারফেক্ট করুন।")
                
                # Pixlr Style Image Enhancements Controls
                st.markdown("### 🎨 Pixlr স্টাইল ফটো এনহ্যান্সমেন্ট")
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    chk_autofix = st.checkbox("🪄 AutoFix (Auto Lighting)", value=True)
                with p_col2:
                    chk_autocontrast = st.checkbox("🌓 Auto Contrast (Boost Details)", value=True)
                with p_col3:
                    slider_sharpen = st.slider("🔪 Sharpen Amount (%)", min_value=0, max_value=100, value=100, step=10)

                # Enhance image using Pixlr logic
                enhanced_base = apply_pixlr_enhancements(
                    st.session_state["clean_rgb"],
                    do_auto_contrast=chk_autocontrast,
                    sharpen_pct=slider_sharpen,
                    do_autofix=chk_autofix
                )

                st.markdown("### 🎛️ লাইভ লাইন কন্ট্রোলার")
                
                col_slider1, col_slider2 = st.columns(2)
                with col_slider1:
                    edge_sens = st.slider(
                        "🔍 প্যাটার্ন ও অবজেক্ট ডিটেইলস (Sensitivity)", 
                        min_value=0, 
                        max_value=100, 
                        value=100, 
                        step=1,
                        help="১০০ তে রাখলে যেকোনো ডিজাইন, প্যাটার্ন বা অবজেক্টের সূক্ষ্ম আউটলাইন ডিটেক্ট করবে।"
                    )
                    min_len = st.slider(
                        "✂️ ছোট দাগ মুছুন (Noise Remove)", 
                        min_value=5, 
                        max_value=100, 
                        value=15, 
                        step=1,
                        help="ছোট অবাঞ্ছিত দাগ মুছতে ব্যবহার করুন।"
                    )
                with col_slider2:
                    smooth_val = st.slider(
                        "〰️ লাইন স্মুথনেস (Curve Fitting)", 
                        min_value=0.0000, 
                        max_value=0.0050, 
                        value=0.0005, 
                        step=0.0001, 
                        format="%.4f",
                        help="০.০০০৫ তে রাখলে অবজেক্ট বা ডিজাইনের মূল কার্ভ শেপ সঠিক থাকবে।"
                    )

                # Process contours live
                preview_img, final_contours = extract_and_draw_lines_live(
                    enhanced_base, 
                    edge_sensitivity=edge_sens, 
                    smoothness=smooth_val, 
                    min_line_length=min_len
                )

                with col_p2:
                    st.image(preview_img, caption="লাইভ ট্রেসিং ভিউ (Pixlr Enhanced Lines)", use_container_width=True)

                st.markdown("---")
                if st.button("📐 ২. স্মুথ DXF ফাইলে কনভার্ট করুন", type="primary"):
                    with st.spinner("৩D অ্যাপের উপযোগী স্মুথ DXF তৈরি হচ্ছে..."):
                        file_base_name = getattr(uploaded_file, 'name', 'Snapshot.jpg')
                        output_filename = f"Pixlr_Smooth_{os.path.splitext(file_base_name)[0]}.dxf"
                        
                        valid_lines = export_smooth_dxf(final_contours, output_filename)

                        if valid_lines > 0:
                            with open(output_filename, "rb") as file:
                                st.success(f"🎉 সফলভাবে {valid_lines} টি স্মুথ ভেক্টর কার্ভ তৈরি হয়েছে!")
                                st.download_button("📥 স্মুথ 3D-রেডি DXF ডাউনলোড করুন", data=file, file_name=output_filename, mime="application/dxf")
                        else:
                            st.warning("⚠️ কোনো আউটলাইন পাওয়া যায়নি।")
