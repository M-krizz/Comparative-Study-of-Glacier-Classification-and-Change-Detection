# Comparative Study of Glacier Classification and Change Detection

## Project Overview
This project investigates the detection of glacial density variations and temporal changes in the **Siachen Glacier** (Eastern Karakoram) using time-series satellite imagery and Google Earth Pro data from **2010 to 2024**. 

As glaciers are critical indicators of global climate health, this research focuses on automating the monitoring process to aid in water resource management, sea-level rise prediction, and disaster preparedness for Glacial Lake Outburst Floods (GLOFs).

## Abstract
With rising global temperatures, the structural volatility of glacier masses has become prominent. This study assesses a spectrum of vision-based algorithms—from conventional image processing to contemporary machine learning architectures—to analyze glacier variation. The results demonstrate that the **U-Net architecture** significantly outperforms other methods in accuracy and boundary demarcation.

## Key Features
- **Comparative Analysis:** Evaluation of multiple classification techniques:
  - **Traditional:** Otsu's Thresholding, Adaptive Thresholding, HSV-based Segmentation.
  - **Machine Learning:** K-Means Clustering, Random Forest (RF).
  - **Deep Learning:** U-Net (CNN).
- **Study Area:** Focused on the Siachen Glacier, one of the longest in the Karakoram range (~76 km), known for its high sensitivity to climate changes.
- **Accuracy:** U-Net achieved an impressive accuracy of **93.42%**.
- **Sustainable Development Goals (SDG):** Aligned with SDG 6 (Clean Water), SDG 2 (Zero Hunger), SDG 13 (Climate Action), and SDG 15 (Life on Land).

## Methodology
The research follows a systematic approach:
1. **Data Collection:** High-resolution 4K imagery captured via Google Earth Pro.
2. **Preprocessing:** Utilizing raw imagery to preserve radiometric consistency for temporal analysis.
3. **Segmentation:** Implementing pixel-wise and object-based analysis (OBIA) techniques.
4. **Performance Analysis:** Comparison using IoU, Dice/F1, Precision, Recall, and Accuracy.

## Results Summary (Siachen Glacier)
| Metric | Adaptive | Otsu | K-Means | HSV | U-Net | RF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 80.62 | 84.83 | 71.09 | 83.16 | **93.42** | 87.81 |
| **F1-Score** | 86.79 | 89.16 | 77.80 | 87.78 | **94.76** | 87.00 |

## Authors
- **Mohana Krishnan M V**
- Sanjay Anand M
- Sabarish B A
- Aarthi R

*Department of Computer Science and Engineering, Amrita School of Computing, Coimbatore, India.*

## Repository Structure
- `Code/`: Implementation scripts for various algorithms (U-Net, RF, Thresholding, etc.).
- `New datas/`: Recent datasets and trained model files.
- `Old dataa/`: Historical samples and reference documents.
- `output/`: Generated masks and comparative results.

---
*This research neccesiates the study of glacier variations in climatic sensitive areas to mitigate the risks associated with global warming.*
