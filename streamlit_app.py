import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Custom title/icon
im = Image.open("App_Icon.jpg")  # cute squid
st.set_page_config(page_title="Colony Counter v1", page_icon=im)

# Header
st.title("🧫 Colony Counter v1")

model = YOLO("weights.pt")  # load weights

# Upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert uploaded file to OpenCV img
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
    
    if st.button("Run YOLO Inference"):
        results = model(img)
        img_annotated = img.copy()
        
        # Converted tensor to iterable numpy array
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Draw only the little green bboxes
        for box in yolo_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Count colonies
        colony_count = len(yolo_boxes)
        
        # Add count in bottom-right corner
        text = f"Colonies: {colony_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3  # text size
        thickness = 5  # text thickness
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img_annotated.shape[1] - text_size[0] - 10  # 10 px from right
        text_y = img_annotated.shape[0] - 10  # 10 px from bottom
        cv2.putText(img_annotated, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
        
        # Save annotated img to file for download
        save_path = "annotated_streamlit.jpg"
        cv2.imwrite(save_path, img_annotated)
        st.success(f"Annotated image saved as {save_path}")
        st.download_button(
            label="Download Annotated Image",
            data=open(save_path, "rb").read(),
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )
        
        # Now, allow adding dots on the annotated image
        st.subheader("Add Manual Dots")
        
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
        
        # HTML and JavaScript for canvas
        html_code = f"""
        <canvas id="canvas" width="{width}" height="{height}" style="border:1px solid #000000;"></canvas>
        <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var img = new Image();
        img.src = 'data:{mime_type};base64,{base64_img}';
        img.onload = function() {{
            ctx.drawImage(img, 0, 0, {width}, {height});
        }};
        canvas.addEventListener('click', function(event) {{
            var rect = canvas.getBoundingClientRect();
            var x = event.clientX - rect.left;
            var y = event.clientY - rect.top;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
        }});
        </script>
        """
        
        # Render the HTML in Streamlit
        st.components.v1.html(html_code, height=height + 10, width=width + 10)
