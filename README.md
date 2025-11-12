# 🧫 Terry the Colony Counter (v1)
Quantify bacterial colonies with automated YOLO-based object detection and manual human-in-the-loop corrections.

---

##  Usage Instructions

1. **Upload your image**  
   Upload the Petri dish image you want to analyze (`.jpg`, `.jpeg`, or `.png`).  
   Once uploaded, your original image will be displayed.

2. **Run YOLO inference**  
   Scroll down and click **`Run YOLO Inference`** to automatically detect and count colonies.  

   You can expand the **`YOLO Detection Options`** dropdown to adjust parameters:
   - **Confidence Threshold:** Filters detections below a set confidence score.  
     - Recommended to keep at **0.0** (minimizes false negatives).  
     - Only raise this if you observe many false positives.
   - **Show Confidence Values:** Displays YOLO’s confidence score for each detected colony (optional).

3. **View results**  
   After running inference, Terry will display:
   - The annotated image with detected colonies  
   - The **total colony count** automatically computed by the model  

4. **Manual adjustment (Human-in-the-Loop correction)**  
   Scroll to the **`Manual Adjustment`** section.  
   This lets you correct any missed detections:
   - Click anywhere on the image to add a red dot where a colony was missed.  
   - Use **`Undo Last Dot`** or **`Clear All Dots`** to remove mistakes.  
   - The **Total Colonies** count updates automatically to include both model detections and manual additions.

---

## 💡 Imaging Recommendations

To get the most accurate results:

- **Colony size:**  
  Works best on **larger, less dense** colonies.  
  → Dilute your sample more and allow colonies to grow larger if possible.

- **Lighting:**  
  Place the Petri dish **lid-side-up** on a bright LED light pad (max brightness).  
  Work in a **dark room** to reduce glare.

- **Focus:**  
  Remove the Petri dish lid before photographing.  
  Make sure colonies are **in sharp focus**.

- **Camera quality:**  
  A modern smartphone camera works great!  
  - For iPhones, use **Portrait → Contour Lighting** mode for best results.
 





🧫 Terry the Colony Counter (v1):
Quantify bacterial colonies with automated YOLO-based object detection and manual human-in-the-loop corrections.

Usage Instructions:
1. upload the image you would like to run inference on (in `.jpg`, `.jpeg`, or `.png` form). you'll be able to see your original uploaded image once it has fully uploaded.
2. by scrolling down past your original image, you will see a `Run YOLO inference` button. this will run the model to automatically detect and quantify colonies in the image. the `YOLO Detection Options` dropdown menu allows you to adjust advanced settings.
    -  `confidence threshold` = will filter out any detections below whatever confidence level you set. not recommended to mess with this setting because the v1 tends to lean towards false negatives. this is only recommend moving it if you're seeing a lot of false positives for some reason. otherwise, keep it set to the default of 0.0 bc the default settings were put in place for a reason
    -  `show confidence values` = will display the model's calculated confidence score for each detected colony. not really necessary
4. once you've ran YOLO inference, it will display an annotated image with the YOLO model's detected colonies and a total count
5. scrolling down, it will display a `Manual Adjustment` section that contains an interactive version of the YOLO annotations. click anywhere on this image where the model missed a colony and it will add a red dot onto the image and manually adjust the total count. press the `Undo Last Dot` or `Clear All Dots` buttons underneath if necessary. when you're done adding any missed colonies, `Total colonies` is displayed underneath the image and this is a sum of the model's auto detected colonies PLUS your manually added colonies.

Imaging Suggestions:
- the model runs better on larger, less dense colonies. diluting the sample more and allowing the colonies to overgrow to a larger size is recommended
- place the petri dish lid-side-up on an LED light pad at max setting in as dark of a room as possible. ensure there is no overhead lighting to minimize glare.
- remove the lid of the petri dish, ensure it is in focus, and use a good camera to take the picture (a smartphone camera works great!)
- if using an iPhone, the "Portrait > Contour Lighting" setting captures the best quality photos
