import cv2
import numpy as np
import gradio as gr

# Farklı filtre fonksiyonları
def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

def apply_invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def apply_fall_filter(frame):
    fall_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return cv2.transform(frame, fall_filter)

def apply_cartoon_filter(frame):
    # Görüntüyü gri tona dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Kenarları yumuşatmak için bulanıklaştır
    gray = cv2.medianBlur(gray, 5)
    # Kenar tespiti için threshold uygula
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    # Renkli resmi yumuşatmak için bilateral filtre uygula
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    # Kenar maskesi ve renkli görüntüyü birleştir
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_vignette_filter(frame, intensity=0.5):
    rows, cols = frame.shape[:2]
    # Gaussian Kernel oluştur
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols * intensity)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows * intensity)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(frame)
    # Vignette etkisini her bir renk kanalına uygula
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette

def apply_warm_filter(frame):
    warm_filter = np.array([[1.2, 0.2, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.8]])
    warm = cv2.transform(frame, warm_filter)
    # Renkleri sınırlayarak aşırı parlaklık ve karanlıkları önle
    return np.clip(warm, 0, 255).astype(np.uint8)

def apply_cold_filter(frame):
    cold_filter = np.array([[0.8, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.2, 1.2]])
    cold = cv2.transform(frame, cold_filter)
    return np.clip(cold, 0, 255).astype(np.uint8)


# Filtre uygulama fonksiyonu
def apply_filter(filter_type, input_image=None):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Web kameradan görüntü alınamadı"

    if filter_type == "Gaussian Blur":
        return apply_gaussian_blur(frame)
    elif filter_type == "Sharpen":
        return apply_sharpening_filter(frame)
    elif filter_type == "Edge Detection":
        return apply_edge_detection(frame)
    elif filter_type == "Invert":
        return apply_invert_filter(frame)
    elif filter_type == "Brightness":
        return adjust_brightness_contrast(frame, alpha=1.0, beta=50)
    elif filter_type == "Grayscale":
        return apply_grayscale_filter(frame)
    elif filter_type == "Sepia":
        return apply_sepia_filter(frame)
    elif filter_type == "Sonbahar":
        return apply_fall_filter(frame)
    elif filter_type == "Vignette":
        return apply_vignette_filter(frame)
    elif filter_type == "Warm":
        return apply_warm_filter(frame)
    elif filter_type == "Cold":
        return apply_cold_filter(frame)
    elif filter_type == "Cartoon":
        return apply_cartoon_filter(frame)
# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Fotoğraf Filtreleme")

    with gr.Row():  # Filtre ve görüntü yükleme alanını yan yana yerleştir
        # Filtre seçenekleri
        filter_type = gr.Dropdown(
            label="Filtre Seçin",
            choices=["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Brightness", "Grayscale", "Sepia", "Sonbahar","Vignette","Warm","Cold","Cartoon"],
            value="Gaussian Blur"
        )

        # Görüntü yükleme alanı
        input_image = gr.Image(label="Resim Yükle", type="numpy")

    # Çıktı için görüntü
    output_image = gr.Image(label="Filtre Uygulandı")

    # Filtre uygula butonu
    apply_button = gr.Button("Filtreyi Uygula")

    # Butona tıklanınca filtre uygulama fonksiyonu
    apply_button.click(fn=apply_filter, inputs=[filter_type, input_image], outputs=output_image)

# Arayüzü başlat
demo.launch()