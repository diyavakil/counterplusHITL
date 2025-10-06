import streamlit as st
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Failed to import YOLO: {e}. Ensure ultralytics and opencv-python-headless are installed.")
    st.stop()
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# custom title/icon
try:
    im = Image.open("App_Icon.jpg")  # cute squid
    st.set_page_config(page_title="Colony Counter v1", page_icon=im, layout="centered")
except FileNotFoundError:
    st.set_page_config(page_title="Colony Counter v1", layout="centered")

# header
st.title("🧫 Colony Counter v1")
st.markdown("""
Quantify bacterial colonies with automated YOLO detection and manual human-in-the-loop corrections. Navigate to the GitHub repository using the button in the top right corner for detailed guide.
""")

try:
    model = YOLO("weights.pt")  # load weights
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}. Ensure the weights file 'weights.pt' is in the correct directory.")
    st.stop()

# upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
if uploaded_file is not None:
    try:
        with st.spinner("Loading image..."):
            # convert uploaded file to OpenCV img
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image.")
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        # options for YOLO
        with st.expander("YOLO Detection Options", expanded=True):
            conf_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Filter out detections below this confidence level.")
            show_conf = st.checkbox("Show confidence values", value=True, help="Display confidence scores for each detected colony.")
        
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
                        cv2.putText(img_annotated, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Count colonies (after filtering)
                colony_count = len(filtered_boxes)
                
                # Add auto count in bottom-right corner
                text = f"Auto Colonies: {colony_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 5
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = img_annotated.shape[1] - text_size[0] - 10
                text_y = img_annotated.shape[0] - 10
                cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                
                # Store annotated image in session state
                st.session_state['img_annotated'] = img_annotated
                st.session_state['colony_count'] = colony_count
                
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
                
                # Manual adjustments section
                st.subheader("Manual Adjustments")
                st.info("Click on the image to add red dots for missed colonies. Use the buttons to manage your edits.")
                
                # Maximum dimensions for display
                MAX_WIDTH = 1920
                MAX_HEIGHT = 1080
                
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
                
                # Convert resized image to bytes for base64 encoding
                buffered = BytesIO()
                resized_img.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode()
                
                mime_type = "image/png"
                
                # HTML and JavaScript for canvas with controls
                html_code = f"""
                <div style="text-align: center;">
                    <canvas id="canvas" width="{width}" height="{height}" style="max-width: 100%; height: auto; border:1px solid #000000; display: block; margin: 0 auto;"></canvas>
                    <div id="count" style="margin: 10px; padding: 10px; background-color: white; color: black; border: 1px solid #ccc; border-radius: 5px; display: inline-block;">
                        Manual additions: 0 | Total colonies: {colony_count}
                    </div>
                    <div style="margin-top: 10px;">
                        <button id="undo" title="Remove the last red dot added" style="margin: 5px; padding: 8px 16px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;">Undo Last Dot</button>
                        <button id="clear" title="Remove all red dots, keeping current image" style="margin: 5px; padding: 8px 16px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;">Clear All Dots</button>
                        <button id="download" title="Download the image with all annotations" style="margin: 5px; padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px;">Download Edited Image</button>
                    </div>
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
                    ctx.imageSmoothingEnabled = true;
                    ctx.imageSmoothingQuality = 'high';
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
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
                    var scaleX = canvas.width / rect.width;
                    var scaleY = canvas.height / rect.height;
                    var x = (event.clientX - rect.left) * scaleX;
                    var y = (event.clientY - rect.top) * scaleY;
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
                
                # Render the HTML in Streamlit within a centered column
                col1, col2, col3 = st.columns([1, 6, 1])  # Adjust column ratios to center content
                with col2:
                    st.components.v1.html(html_code, height=height + 150)
            
    except Exception as e:
        st.error(f"Error processing image: {e}. Try uploading a different image or checking the file format.")
