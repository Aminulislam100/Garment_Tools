import os
import time
import cv2
import ezdxf
import numpy as np
import streamlit as st
from rembg import remove
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ==========================================
# গ্লোবাল পেজ কনফিগারেশন
# ==========================================
st.set_page_config(page_title="AI Smart Suite", layout="wide", page_icon="🚀")

# ==========================================
# Top Navigation Bar Styling (CSS & UI)
# ==========================================
st.markdown("""
<style>
/* Top Nav Bar Styling */
.stRadio > div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    background: #0f172a;
    padding: 10px;
    border-radius: 12px;
    gap: 15px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.stRadio > div[role="radiogroup"] > label {
    background-color: #1e293b;
    color: #f8fafc !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    font-weight: 600 !important;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stRadio > div[role="radiogroup"] > label:hover {
    background-color: #334155;
    border-color: #3b82f6 !important;
}
.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border-color: #60a5fa !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Top Navbar Menu
app_mode = st.radio(
    "Main Navigation",
    ["🧵 Fabric Vision AI", "📐 DXF Converter"],
    horizontal=True,
    label_visibility="collapsed"
)

# ==========================================
# গ্লোবাল ফাংশনসমূহ (Fabric Vision AI এর জন্য)
# ==========================================
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
# পেজ ১: Fabric Vision AI
# ==========================================
if app_mode == "🧵 Fabric Vision AI":
    
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

    st.sidebar.header("⚙️ Fabric Vision Settings")
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

    st.markdown("""
    <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 16px 22px; border-radius: 14px 14px 0 0; color: white; font-weight: bold; font-size: 18px; box-shadow: 0 6px 15px rgba(4,120,87,0.35); border: 6px solid #047857; border-bottom: none;">
        🔬 ধাপ ২: টেস্টিং স্ক্যানার (Production Check)
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
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
                        if cosine_sim < 0.75:
                            pattern_pct = 0.0
                        else:
                            pattern_pct = ((cosine_sim - 0.75) / (1.0 - 0.75)) * 100.0
                        
                        lap_diff = abs(b_lap - cam_lap)
                        texture_pct = max(0.0, 100.0 - (lap_diff / (max(b_lap, 1e-5)) * 150.0))
                        
                        if "শুধুমাত্র কালার/শেডিং" in inspection_mode:
                            final_score = color_pct
                        elif "শুধুমাত্র ডিজাইন/প্রিনট" in inspection_mode:
                            final_score = pattern_pct
                        elif "সুতার ঘনত্ব / টেক্সচার" in inspection_mode:
                            final_score = texture_pct
                        elif "কালার + ডিজাইন" in inspection_mode:
                            final_score = (color_pct * 0.50) + (pattern_pct * 0.50)
                        else:  
                            final_score = (color_pct * 0.40) + (pattern_pct * 0.40) + (texture_pct * 0.20)
                        
                        if final_score > best_match_score:
                            best_match_score = final_score
                            best_match_path, best_match_name = b_path, name
                            d_color, d_pattern, d_texture = color_pct, pattern_pct, texture_pct
                
                    st.write(f"**চেকিং মোড:** `{inspection_mode}`")
                    st.markdown(f"### **ফাইনাল একুরেসি:** `{best_match_score:.2f}%`")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("🎨 Color/Shade Score", f"{d_color:.1f}%")
                    col_m2.metric("✨ AI Vector Pattern", f"{d_pattern:.1f}%")
                    col_m3.metric("🧶 Texture Density", f"{d_texture:.1f}%")
                    
                    if best_match_score >= pass_threshold:
                        st.success("### 🎉 PASS - প্রোডাক্ট কোয়ালিটি সঠিক আছে!")
                    else:
                        st.error("### ❌ FAIL - রিজেক্টেড! (পার্থক্য পাওয়া গেছে)")
                        
                    st.markdown("---")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), caption="আপনার স্ক্যান (Test)", use_container_width=True)
                    with v_col2:
                        if best_match_path:
                            st.image(cv2.cvtColor(cv2.imread(best_match_path), cv2.COLOR_BGR2RGB), caption=f"ম্যাচিং মাস্টার: {best_match_name}", use_container_width=True)

# ==========================================
# পেজ ২: DXF Converter
# ==========================================
elif app_mode == "📐 DXF Converter":
    
    # সাব-মেনু (শুধুমাত্র DXF পেজ সিলেক্ট করলে দেখাবে)
    st.sidebar.markdown("### 📐 DXF Options")
    dxf_page = st.sidebar.radio("আপনার প্রয়োজনীয় টুলটি বেছে নিন:", ["Convert Clear Photo", "Process Photo and Convert"])
    st.sidebar.info("১. **Convert Clear Photo:** শুধু পরিষ্কার ছবির জন্য দ্রুত কনভার্টার।\n২. **Process Photo:** অস্পষ্ট ছবির ব্যাকগ্রাউন্ড রিমুভ ও লাইন শার্প করার অ্যাডভান্সড টুল।")

    if dxf_page == "Convert Clear Photo":
        st.title("📐 Convert Clear Photo")
        st.write("আপনার জ্যামিতিক বা টাইলসের পরিষ্কার ছবি আপলোড করুন, এক ক্লিকে DXF ডাউনলোড করুন।")

        uploaded_file = st.file_uploader("একটি ছবি বাছাই করুন", type=["jpg", "jpeg", "png", "bmp"], key="page1_uploader")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                st.image(img, caption="আপলোড করা ছবি", width=300)

                if st.button("⚡ DXF ফাইলে কনভার্ট করুন", key="btn1"):
                    with st.spinner("প্রসেসিং হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন..."):
                        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        doc = ezdxf.new(dxfversion="R2010")
                        msp = doc.modelspace()

                        for cnt in contours:
                            if len(cnt) > 2:
                                points = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
                                msp.add_lwpolyline(points, close=True)

                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}.dxf"
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

    elif dxf_page == "Process Photo and Convert":
        st.title("⚙️ Process Photo and Convert")
        st.write("এই টুলটি স্বয়ংক্রিয়ভাবে ব্যাকগ্রাউন্ড রিমুভ, অটো কন্ট্রাস্ট এবং শার্প করে নিখুঁত DXF তৈরি করবে।")

        uploaded_file = st.file_uploader("একটি ছবি বাছাই করুন", type=["jpg", "jpeg", "png", "bmp"], key="page2_uploader")

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="অরিজিনাল ছবি", width=300)

            if st.button("⚡ প্রসেস ও DXF ফাইলে কনভার্ট করুন", key="btn2"):
                with st.spinner("অ্যাডভান্সড প্রসেসিং হচ্ছে (ব্যাকগ্রাউন্ড রিমুভ ও শার্পেন)... দয়া করে অপেক্ষা করুন।"):
                    try:
                        # ১. AI দিয়ে ব্যাকগ্রাউন্ড রিমুভ করা
                        img_no_bg = remove(input_image)
                        img_array = np.array(img_no_bg)
                        
                        # ২. ট্রান্সপারেন্ট ব্যাকগ্রাউন্ডকে সাদা করা (যাতে লাইন কালো হয়)
                        if img_array.shape[2] == 4:
                            alpha_channel = img_array[:, :, 3]
                            rgb_channels = img_array[:, :, :3]
                            white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
                            alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
                            alpha_factor = np.concatenate([alpha_factor, alpha_factor, alpha_factor], axis=2)
                            img_rgb = rgb_channels * alpha_factor + white_background * (1 - alpha_factor)
                            img_rgb = img_rgb.astype(np.uint8)
                        else:
                            img_rgb = img_array

                        # ৩. গ্রেস্কেল (সাদাকালো) রূপান্তর
                        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

                        # ৪. অটো কন্ট্রাস্ট (CLAHE) - হালকা লাইন স্পষ্ট করার জন্য
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced_gray = clahe.apply(gray)

                        # ৫. শার্পেনিং (Sharpening Filter) - ব্লার বা ঝাপসা কমানোর জন্য
                        kernel = np.array([[-1, -1, -1],
                                           [-1,  9, -1],
                                           [-1, -1, -1]])
                        sharpened = cv2.filter2D(enhanced_gray, -1, kernel)
                        
                        st.image(sharpened, caption="ক্লিন ও শার্প করা ছবি (ট্রেসিংয়ের জন্য প্রস্তুত)", width=300)

                        # ৬. ট্রেসিং (Thresholding & Contours)
                        _, thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # ৭. DXF ফাইল তৈরি
                        doc = ezdxf.new(dxfversion='R2010')
                        msp = doc.modelspace()

                        for cnt in contours:
                            # অতিরিক্ত ছোট নয়েজ বা ডট বাদ দেওয়া
                            if cv2.contourArea(cnt) > 10: 
                                points = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
                                msp.add_lwpolyline(points, close=True)

                        # ৮. আউটপুট ফাইল সেভ ও ডাউনলোড
                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_Processed.dxf"
                        doc.saveas(output_filename)

                        with open(output_filename, "rb") as file:
                            st.success("সফলভাবে নিখুঁত DXF তৈরি হয়েছে!")
                            st.download_button(
                                label="📥 ফ্রেশ DXF ফাইল ডাউনলোড করুন",
                                data=file,
                                file_name=output_filename,
                                mime="application/dxf",
                                key="dl_btn2"
                            )
                    except Exception as e:
                        st.error(f"প্রসেস করার সময় একটি সমস্যা হয়েছে: {e}")
