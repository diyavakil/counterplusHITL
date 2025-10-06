import streamlit as st
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Failed to import YOLO: {e}. Please ensure ultralytics and opencv-python-headless are installed.")
    st.stop()
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Custom title/icon
try:
    im = Image.open("App_Icon.jpg")  # cute squid
    st.set_page_config(page_title="Colony Counter v1", page_icon=im)
except FileNotFoundError:
    st.set_page_config(page_title="Colony Counter v1")

# Header
st.title("🧫 Colony Counter v1")
st.markdown("Upload an image of bacterial colonies, run YOLO detection, and manually add dots for any missed colonies. The app will provide an updated total count dynamically.")

try:
    model = YOLO("weights.pt")  # load weights
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}. Please check the weights file.")
    st.stop()

# Upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Convert uploaded file to OpenCV img
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        # Options for YOLO
        with st.expander("YOLO Detection Options"):
            conf_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01, help="Filter detections below this confidence level.")
            show_conf = st.checkbox("Show confidence values", value=False, help="Display confidence scores next to each detected colony.")
        
        if st.button("Run YOLO Inference"):
            with st.spinner("Running YOLO inference..."):
                results = model(img)
                img_annotated = img.copy()
                
                # Get boxes and confidences
                yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
                yolo_confs = results[0].boxes.conf.cpu().numpy()
                
                # Filter by threshold
                mask = yolo_confs >= conf_threshold
                filtered_boxes = yolo_boxes[mask]
                filtered_confs = yolo_confs[mask]
                
                # Draw green boxes (and confidences if enabled)
                for i, box in enumerate(filtered_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    if show_conf:
                        conf_text = f"{filtered_confs[i]:.2f}"
                        cv2.putText(img_annotated, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Count colonies (after filtering)
                colony_count = len(filtered_boxes)
                
                # Add auto count in bottom-right corner
                text = f"Auto Colonies: {colony_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3  # text size
                thickness = 5  # text thickness
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = img_annotated.shape[1] - text_size[0] - 10  # 10 px from right
                text_y = img_annotated.shape[0] - 10  # 10 px from bottom
                cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                
                st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image (Auto Detections)", use_column_width=True)
                
                # Save annotated img to file for download
                save_path = "annotated_streamlit.jpg"
                cv2.imwrite(save_path, img_annotated)
                with open(save_path, "rb") as f:
                    st.download_button(
                        label="Download Annotated Image (Auto Only)",
                        data=f,
                        file_name="annotated_image_auto.jpg",
                        mime="image/jpeg"
                    )
                
                # Now, allow adding dots on the annotated image
                st.subheader("Manual Adjustments")
                st.info("Click on the image below to add red dots for missed colonies. Use the buttons to undo, clear, or download the edited image. The counts update automatically.")
                
                # Maximum dimensions for display (increased for better quality)
                MAX_WIDTH = 1200
                MAX_HEIGHT = 800
                
                def resize_image(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
                    """Resize image while preserving aspect ratio."""
                    img = image.copy()
                    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    return img
                
                # Convert annotated OpenCV image to PIL
                img_pil = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
                
                # Resize image to fit within max dimensions
                resized_img = resize_image(img_pil)
                width, height = resized_img.size
                
                # Convert resized image to bytes for base64 encoding, using PNG for lossless quality
                buffered = BytesIO()
                resized_img.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode()
                
                # Use PNG MIME type for better quality
                mime_type = "image/png"
                
                # HTML and JavaScript for canvas with controls
                html_code = f"""
                <div style="text-align: center;">
                    <canvas id="canvas" width="{width}" height="{height}" style="border:1px solid #000000; display: block; margin: 0 auto;"></canvas>
                    <div id="count" style="margin: 10px;">Manual additions: 0 | Total colonies: {colony_count}</div>
                    <button id="undo" style="margin: 5px;">Undo Last Dot</button>
                    <button id="clear" style="margin: 5px;">Clear All Dots</button>
                    <button id="download" style="margin: 5px;">Download Edited Image</button>
                </div>
                <script>
                var canvas = document.getElementById('canvas');
                var ctx = canvas.getContext('2d');
                var img = new Image();
                img.src = 'data:{mime_type};base64,{base64_img}';
                var points = [];
                var initial_count = {colony_count};
                function redraw() {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, {width}, {height});
                    for (var p of points) {{
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
                        ctx.fillStyle = 'red';
                        ctx.fill();
                    }}
                    var manual = points.length;
                    var total = initial_count + manual;
                    document.getElementById('count').innerText = 'Manual additions: ' + manual + ' | Total colonies: ' + total;
                }}
                img.onload = function() {{
                    redraw();
                }};
                canvas.addEventListener('click', function(event) {{
                    var rect = canvas.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    points.push({{x: x, y: y}});
                    redraw();
                }});
                document.getElementById('undo').addEventListener('click', function() {{
                    points.pop();
                    redraw();
                }});
                document.getElementById('clear').addEventListener('click', function() {{
                    points = [];
                    redraw();
                }});
                document.getElementById('download').addEventListener('click', function() {{
                    var link = document.createElement('a');
                    link.download = 'edited_image.png';
                    link.href = canvas.toDataURL('image/png');
                    link.click();
                }});
                </script>
                """
                
                # Render the HTML in Streamlit
                st.components.v1.html(html_code, height=height + 100, width=width + 20)
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
