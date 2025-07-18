#  Potato Disease Classifier

A deep learning-based classification system that detects and identifies common diseases in potato leaves—**Early Blight**, **Late Blight**, and **Healthy**—using convolutional neural networks (CNNs). This project aims to support the agricultural sector through rapid, scalable, and automated disease diagnosis.

---

##  Motivation

Potato is the **fourth most important food crop globally**, following rice, wheat, and maize. Diseases like **Early Blight** and **Late Blight** are devastating fungal infections that drastically reduce crop yield and quality. Traditionally, disease diagnosis is manual—prone to **human error**, time-consuming, and limited by expert availability. This project demonstrates the power of machine learning in **bridging the knowledge gap** in rural farming and **reducing economic losses** due to late or incorrect disease identification.

---

##  Features

-  Detects Early Blight, Late Blight, and Healthy leaves using image classification
-  Trained using CNNs with TensorFlow and Keras
-  Supports high-resolution RGB image input
-  Includes performance metrics and visualizations
-  Modular design for future integration with mobile/IoT devices

---

##  About the Diseases

### 1. **Early Blight** *(Alternaria solani)*
- Symptoms: Small brown spots with concentric rings
- Effect: Reduces photosynthesis, leading to reduced tuber size
- Common in dry, warm climates

### 2. **Late Blight** *(Phytophthora infestans)*
- Symptoms: Water-soaked lesions on leaves and stems
- Effect: Entire crop failure if untreated; caused the historic Irish Potato Famine
- Thrives in cool, wet conditions

---

##  Real-World Importance

-  **Impact on Smallholder Farmers**: Rapid detection enables timely pesticide intervention, saving crops.
-  **Economic Savings**: Prevents yield losses of up to 50–70% for infected farms.
-  **Sustainable Agriculture**: Reduces excessive pesticide use by enabling precision agriculture.
-  **Research Utility**: Provides a foundation for plant pathology research, dataset augmentation, and real-time diagnosis tools.

---

##  Project Structure

```

potato\_disease\_classifier/
├── dataset/                   # Image dataset (organized by class)
├── train.py                   # Train the CNN model
├── predict.py                 # Predict disease from an image
├── utils.py                   # Helper functions
├── potato\_disease\_model.h5    # Trained model
├── requirements.txt           # Required Python libraries
└── README.md

````

---

##  Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/potato-disease-classifier.git
cd potato-disease-classifier
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Organize your dataset:

```
dataset/
├── Early_Blight/
├── Late_Blight/
└── Healthy/
```

4. Train the model:

```bash
python train.py
```

5. Predict from a new image:

```bash
python predict.py path_to_leaf_image.jpg
```

---

##  Sample Prediction

```
Prediction: Early Blight (96.42%)
```

---

##  Future Improvements

*  **Mobile App** using TensorFlow Lite or React Native for real-time leaf scanning
*  **Live Detection** using OpenCV and edge devices (e.g., Raspberry Pi with a camera)
*  **Multilingual Farmer App** for better accessibility
*  **Disease Severity Estimation** (mild, moderate, severe)
*  **Extension to Other Crops** like tomatoes, peppers, and cucumbers

---

##  Limitations

* Dataset biases (e.g., controlled lighting) may affect real-world performance
* Leaf images must be clear and close-up for optimal accuracy
* Does not estimate disease severity (currently a binary classification)

---

##  References

* [PlantVillage Dataset – Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
* [Phytopathology Research Journal](https://phytopathology.org/)
* "Deep Learning for Plant Disease Detection" – Mohanty, S. P. et al., 2016

---

##  License

This project is licensed under the **MIT License**—free to use, share, and modify.

---

##  Contributing

Pull requests and suggestions are welcome. Whether it's fixing bugs, adding features, or proposing new ideas—your curiosity is welcome here.

---

##  Curiosity Drives Innovation

This project began as a question: *"Can AI see what farmers see?"* From that spark, this classifier emerged as a tool for transforming how we approach disease management in agriculture—making precision farming more accessible and data-driven. 



