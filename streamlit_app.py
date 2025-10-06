```python
import streamlit as st
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Failed to import YOLO: {e}. Ensure ultralytics and opencv-python-headless are installed in your environment.")
    st.stop()
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Custom title/icon
try:
    im = Image.open("App_Icon.jpg")  # cute squid
    st.set_page_config(page_title="Colony Counter v1", page_icon=im, layout="centered")
except FileNotFoundError:
    st.set_page_config(page_title="Colony Counter v1", layout="centered")

# Sidebar
st.sidebar.title("Colony Counter v1")
st.sidebar.markdown("""
**Version**: 1.0  
**Description**: Upload a bacterial colony image, run YOLO detection, and manually add dots for missed colonies. Adjust confidence thresholds and toggle confidence display as needed.  
**Instructions**:  
1. Upload a JPG, JPEG, or PNG image.  
2. Set YOLO options and click "Run YOLO Inference".  
3. Click on the annotated image to add red dots.  
4. Use buttons to undo, clear, or download the edited image.
""")

# Header
st.title("🧫 Colony Counter v1")
st.markdown("Analyze bacterial colonies with automated YOLO detection and manual dot additions for precise counting.")

try:
    model = YOLO("weights.pt")  # load weights
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}. Ensure the weights file 'weights.pt' is in the correct directory.")
    st.stop()

# Upload img
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
if uploaded_file is not None:
    try:
        with st.spinner("Loading image..."):
            # Convert uploaded file to OpenCV img
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image.")
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        # Options for YOLO
        with st.expander("YOLO Detection Options", expanded=True):
            conf_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Filter detections below this confidence level.")
            show_conf = st.checkbox("Show confidence values", value=True, help="Display confidence scores next to each detected colony.")
        
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
                
                # Store annotated image in session state for reset
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
                
                # Convert resized image to bytes for base64 encoding
                buffered = BytesIO()
                resized_img.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode()
                
                mime_type = "image/png"
                
                # HTML and JavaScript for canvas with controls
                html_code = f"""
                <div style="text-align: center;">
                    <canvas id="canvas" width="{width}" height="{height}" style="border:1px solid #000000; display: block; margin: 0 auto;"></canvas>
                    <div id="count" style="margin: 10px; padding: 10px; background-color: white; color: black; border: 1px solid #ccc; border-radius: 5px; display: inline-block;">
                        Manual additions: 0 | Total colonies: {colony_count}
                    </div>
                    <div style="margin-top: 10px;">
                        <button id="undo" style="margin: 5px; padding: 8px 16px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;">Undo Last Dot</button>
                        <button id="clear" style="margin: 5px; padding: 8px 16px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;">Clear All Dots</button>
                        <button id="reset" style="margin: 5px; padding: 8px 16px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;">Reset to Auto</button>
                        <button id="download" style="margin: 5px; padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px;">Download Edited Image</button>
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
                document.getElementById('reset').addEventListener('click', function() {{
                    points = [];
                    img.src = 'data:{mime_type};base64,{base64_img}';
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
                st.components.v1.html(html_code, height=height + 150, width=width + 20)
                
                # Reset to auto-annotated image
                if st.button("Reset to Auto-Annotated Image"):
                    if 'img_annotated' in st.session_state:
                        st.image(cv2.cvtColor(st.session_state['img_annotated'], cv2.COLOR_BGR2RGB), caption="Annotated Image (Auto Detections)", use_column_width=True)
                        with open(save_path, "rb") as f:
                            st.download_button(
                                label="Download Annotated Image (Auto Only)",
                                data=f,
                                file_name="annotated_image_auto.jpg",
                                mime="image/jpeg"
                            )
                        st.components.v1.html(html_code, height=height + 150, width=width + 20)
            
    except Exception as e:
        st.error(f"Error processing image: {e}. Try uploading a different image or checking the file format.")
```

### Explanation of Changes
1. **Confidence Threshold Default**:
   - Changed `conf_threshold` slider default from `0.25` to `0.00` to show all detections by default, as requested.
   - Kept the step size at `0.01` for fine-grained control.

2. **Text Visibility in Dark Mode**:
   - Modified the HTML `<div>` for the count display to include `style="background-color: white; color: black; border: 1px solid #ccc; border-radius: 5px;"`.
   - This ensures the "Manual additions: X | Total colonies: Y" text is readable in both light and dark modes by using a white background with black text and a subtle border.

3. **Confidence Value Style**:
   - Updated the `cv2.putText` call for confidence values to match your Colab code:
     - Font: `cv2.FONT_HERSHEY_SIMPLEX`
     - Font size: `1`
     - Thickness: `2`
     - Position: `(x1, y1 - 5)`
     - Color: `(0, 255, 0)` (green)
   - This makes the confidence values larger and more readable, consistent with your Colab notebook.

4. **Professional and User-Friendly Enhancements**:
   - **Centered Layout**: Used `layout="centered"` in `st.set_page_config` and centered the canvas and buttons using HTML/CSS for a polished look.
   - **Sidebar Instructions**: Added a sidebar with app version, description, and clear instructions to guide users.
   - **Better Feedback**: Added `st.spinner` for image loading and YOLO inference to indicate processing.
   - **Styled Buttons**: Applied CSS to buttons (undo, clear, reset, download) with padding, borders, and a green download button for visual hierarchy.
   - **Reset Button**: Added a "Reset to Auto-Annotated Image" button using `st.session_state` to reload the original YOLO-annotated image without re-running inference.
   - **Improved Error Messages**: Enhanced error messages to be more specific and actionable (e.g., suggesting to check file format).
   - **Expanded Options by Default**: Set the YOLO options expander to `expanded=True` for immediate visibility.
   - **Clear Instructions**: Added `st.info` and `st.markdown` to explain the manual adjustment process clearly.

### Deployment Notes
To ensure the app runs correctly:
1. **Update `requirements.txt`**:
   ```plaintext
   streamlit
   ultralytics
   opencv-python-headless==4.10.0.84
   numpy
   pillow
   ```
2. **Update `packages.txt`** (for Streamlit Cloud):
   ```plaintext
   libgl1-mesa-glx
   libglib2.0-0
   ```
3. **Python Version**: If Python 3.13 causes issues, add a `.python-version` file with:
   ```plaintext
   3.12
   ```
4. **Weights File**: Ensure `weights.pt` is in the project directory and accessible.
5. **Reboot App**: After pushing changes to your repository, reboot the app on Streamlit Cloud.

### Testing
- Upload an image and verify that YOLO detections appear with confidence values (if enabled) in the specified style.
- Check that the confidence threshold slider starts at 0.00.
- Add manual dots and confirm the "Manual additions" and "Total colonies" text is visible in both light and dark modes.
- Test undo, clear, reset, and download buttons.
- Verify that confidence values are readable and match the Colab style.

If you encounter any issues or want further tweaks (e.g., different colors, additional features), let me know!
