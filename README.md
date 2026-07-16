# 🚀 End-to-End Machine Learning & Computer Vision

**Student Name:** Amlan Pal  
**College Roll Number:** 23BCE0562  
**Student Signature:** 02ce8a497ee1  

Welcome to my End-to-End Machine Learning & Computer Vision Project!This repository contains a complete, automated machine learning pipeline that transitions from basic tabular data analysis all the way to deploying a deep learning model for complex computer vision tasks. 

---

### 📊 Datasets Used
To demonstrate versatility across different AI domains, this project utilizes five distinct datasets:
1. **Mobile Price Classification** (Exploratory Data Analysis)
2. **Wine Quality Dataset** (Multi-class Classification)
3. **California Housing Dataset** (Continuous Regression)
4. **Iris Flowers Dataset** (Unsupervised Clustering)
5. **VisDrone Benchmark Dataset** (Advanced Computer Vision / Object Detection)
6. **MNIST Digits** (Secondary CNN Classification)

---

### 🛠️ Tasks Completed
* **Task 1: EDA & Feature Correlation:** Mapped out feature distributions and discovered the heavy mathematical correlation between a phone's RAM and its market price.
* **Task 2: Supervised Classification:** Pitted Logistic Regression against a computationally handicapped Decision Tree.
* **Task 3: Continuous Regression:** Built a Linear Regression model with dynamic, seed-based feature dropping to predict California real estate prices.
* **Task 4: Unsupervised Clustering:** Grouped 4D biological data using K-Means and visualized the results in 2D using Principal Component Analysis (PCA).
* **Task 5: Advanced Model Building:** Deployed a sequential CNN for digit recognition and fine-tuned a pre-trained YOLOv8 Nano architecture on the massive 2GB VisDrone dataset.

> **💡 Note on the Advanced Track:** I chose **Track C (YOLO-Based Object Detection)**. To fulfill the bonus capstone requirements, I bypassed the subset data and trained the model on the *entire* 2GB VisDrone dataset. I also built a secondary CNN model (Track B) to satisfy the extended module constraints.

---

### ⚙️ How to Run the Code

1. **Environment:** Open the provided `AIML_CAPSTONE_AMLANPAL_23BCE0562.ipynb` file in **Google Colab**.
2. **Hardware:** Navigate to `Runtime` -> `Change runtime type` and set the Hardware Accelerator to **T4 GPU** (This is absolutely critical for the YOLOv8 training in Task 5).
3. **Initialization:** Run the very first cell (Cohort Binder) to initialize the personalized `SEED` (47090249). This seed mathematically dictates the hyperparameter constraints and feature limitations for the entire notebook.
4. **Data Uploads:** Ensure `iris_dataset.csv` is uploaded to the root Colab directory (`/content/`), and that the Mobile Price and Wine Quality zip files are available in your mounted Google Drive.
5. **Execution:** Run the cells sequentially. Task 5 will automatically trigger an Ultralytics script to download the 2GB VisDrone dataset on the fly—no manual downloading required!

---

### 📈 Key Results
* **Classification (Wine Quality):** Logistic Regression outperformed the Decision Tree (Accuracy: 65.07% vs 56.77%). The tree's underperformance was an expected mathematical consequence of my personalized SEED capping the `max_depth` at 6, forcing the model to underfit.
* **Regression (California Housing):** Achieved an $R^2$-score of 0.5862. The model successfully identified a dataset quirk where maximum house prices were artificially capped at $500k by the original researchers.
* **Clustering (Iris PCA):** Successfully mapped 4D biological data to a 2D plot. K-Means effectively isolated the *Setosa* species, though my algorithmic constraint of $k=4$ artificially fractured the remaining two species across unnecessary clusters.
* **Advanced Vision (YOLOv8):** Fine-tuned the `yolov8n.pt` model over 30 epochs. Achieved a solid overall mAP@50 of 0.283, successfully drawing accurate bounding boxes around tiny, heavily occluded objects like cars, buses, and pedestrians from complex aerial drone viewpoints.

---

### 🧠 Major Learnings
* **The Impact of Constraints:** Forcing hyperparameter limits (like `max_depth` or a specific `k` cluster value) based on a cryptographic seed was a masterclass in understanding how models behave when they are intentionally handicapped or forced to drop predictive features.
* **Data Scaling is Critical:** Receiving a `ConvergenceWarning` during Logistic Regression reinforced the practical necessity of standardizing distance-based features to balance algorithm weights during gradient descent.
* **Computer Vision Realities:** Detecting objects in VisDrone is vastly harder than standard datasets (like COCO) because the targets are tiny clusters of pixels. I learned how pre-trained weights significantly accelerate edge detection, but handling dense, high-altitude imagery ultimately requires higher training resolutions and techniques like SAHI (Slicing Aided Hyper Inference) to preserve visual data.
