# 🧫 Terry the Colony Counter (v1)
Quantify bacterial colonies with automated YOLO-based object detection and manual human-in-the-loop corrections.
NOTE: The current version (v1) tends to have a decent number of false negative detections. The human-in-the-loop addition is meant to adjust the automatic count to ensure accuracy. All images are highly recommended to be checked manually post-inference to make any necessary corrections.

---

##  Usage Instructions

1. **Upload your image**  
   Upload the Petri dish image you want to analyze (`.jpg`, `.jpeg`, or `.png`). Once uploaded, your original image will be displayed.
2. **Run YOLO inference**  
   Scroll down and click `Run YOLO Inference` to automatically detect and count colonies. Optionally, you can expand the `YOLO Detection Options` dropdown to adjust parameters:
   - **Confidence Threshold:** Filters detections below a set confidence score. Recommended not adjust it and to keep it at the default of 0.0 (minimizes false negatives). Only raise this if you observe many false positives.
   - **Show Confidence Values:** Displays YOLO’s confidence score for each detected colony (optional). The confidence scores showing may make it harder to complete the human-in-the-loop corrections for denser plates (as it does not allow full visibility of potentially missed colonies).
3. **View results**  
   After running inference, Terry will display the annotated image with detected colonies highlighted in green and the total colony count automatically computed by the model in the bottom right corner.
4. **Manual adjustment (Human-in-the-Loop correction)**  
   Scroll to the **`Manual Adjustment`** section. This lets you correct any missed detections:
   - Click anywhere on the image to add a red dot where a colony was missed.
   - Use **`Undo Last Dot`** or **`Clear All Dots`** to remove mistakes.
   - The **Total Colonies** count updates automatically to include both model detections and manual additions.

---

## Imaging Recommendations

These are just suggestions, but the model's ability to efficiently run inference does greatly depend on the quality of the input image.
- **Colony size:** Works best on larger, less dense colonies. It is recommended to dilute your sample more and allow colonies to grow larger if possible.
- **Lighting:** Place the petri dish lid-side-up on a bright LED light pad (max brightness). Work in a dark room to reduce glare.
- **Focus:** Remove the Petri dish lid before photographing. Make sure colonies are in sharp focus.
- **Camera quality:** A smartphone camera works great! For iPhones, use Portrait > Contour Lighting mode for best results.
