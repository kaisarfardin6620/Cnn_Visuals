# CNN Visualizations Toolkit

A comprehensive collection of visualization techniques for Convolutional Neural Networks (CNNs) to understand model behavior, performance, and interpretability.

## 📋 Overview

This Jupyter notebook (`cnn_visuals.ipynb`) provides essential visualization tools for CNN analysis, including data exploration, model architecture visualization, feature interpretation, and performance evaluation. These visualizations help in understanding what your CNN learns and how well it performs on your image classification tasks.

## Features

### 1. Data and Preprocessing Visualizations
- **Sample Image Display**: View original images from your training dataset
- **Data Augmentation Visualization**: See how augmentation techniques transform your images
- **Class Distribution Analysis**: Understand the balance of classes across train/validation/test splits

### 2. Model Architecture Visualizations
- **Model Summary**: Detailed layer-by-layer breakdown of your CNN
- **Architecture Graph**: Visual representation of model structure with layer connections

### 3. Feature and Interpretability Visualizations
- **Activation Maps**: Visualize feature maps from intermediate convolutional layers
- **Filter Visualization**: See what patterns the first convolutional layer detects
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Understand which parts of an image are important for predictions

### 4. Model Performance Visualizations
- **Training History**: Loss and accuracy curves over training epochs
- **Confusion Matrix**: Detailed breakdown of correct and incorrect predictions
- **Classification Report**: Precision, recall, and F1-scores for each class
- **ROC Curves**: Receiver Operating Characteristic curves for multi-class evaluation
- **Prediction Samples**: Visual examples of model predictions with confidence scores

### 5. Model Comparison Visualizations
- **Training History Comparison**: Compare multiple models' learning curves
- **Performance Metrics Comparison**: Side-by-side comparison of final model performance

## Prerequisites

### Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from IPython.display import Image, display
```

### Additional Dependencies
- `graphviz` and `pydot` for model architecture visualization
- `opencv-python` for image processing in Grad-CAM

Install with:
```bash
pip install graphviz pydot opencv-python
```

## Usage

### Prerequisites Setup
Before running the visualizations, ensure you have:

1. A trained CNN model (`model`)
2. Training history object (`history`)
3. Data generators (`train_generator`, `val_generator`, `test_generator`)
4. DataFrame objects (`train_df`, `val_df`, `test_df`)
5. Image dimensions defined (`IMG_HEIGHT`, `IMG_WIDTH`)

### Running the Notebook

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook cnn_visuals.ipynb
   ```

3. **Run cells sequentially** to generate all visualizations

### Key Functions

#### Grad-CAM Implementation
```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for interpretability
    """
```

#### Visualization Display
```python
def display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays heatmap on original image
    """
```

## Visualization Examples

### Data Exploration
- View original and augmented training images
- Analyze class distribution across data splits

### Model Understanding
- Examine learned filters in convolutional layers
- Visualize activation maps at different network depths

### Performance Analysis
- Track training progress with loss/accuracy curves
- Evaluate model performance with confusion matrices
- Compare multiple model architectures

### Interpretability
- Generate Grad-CAM heatmaps to understand model focus areas
- Visualize which image regions contribute to predictions

## Customization

### Adapting for Your Model
1. **Update layer names**: Modify layer selection based on your model architecture
2. **Adjust image preprocessing**: Update normalization ranges based on your preprocessing pipeline
3. **Customize class names**: Update `class_names` list to match your dataset classes
4. **Modify visualization parameters**: Adjust figure sizes, color maps, and display counts

### Example Modifications
```python
# For different normalization ranges
if display_image.min() < 0:  # [-1, 1] range
    display_image = (display_image * 0.5 + 0.5)
elif display_image.max() > 1:  # [0, 255] range
    display_image = display_image / 255.0
```

## Troubleshooting

### Common Issues

1. **GraphViz Installation**: 
   - Ensure GraphViz is installed system-wide and added to PATH
   - On Windows: Download from GraphViz website
   - On Linux: `sudo apt-get install graphviz`
   - On macOS: `brew install graphviz`

2. **Memory Issues**:
   - Reduce batch sizes for visualization
   - Process fewer images at once for large datasets

3. **Layer Name Errors**:
   - Check your model's layer names with `model.summary()`
   - Update layer selection code accordingly

## File Structure

```
├── cnn_visuals.ipynb          # Main notebook with all visualizations
├── README.md                  # This file
├── model_architecture.png     # Generated model architecture diagram
└── requirements.txt           # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

