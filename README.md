# Rice Leaf Disease Classification (MobileNet)

Image classification project detecting three rice leaf conditions: bacterial
leaf blight, brown spot, and leaf smut, using transfer learning on a
MobileNet backbone pretrained on ImageNet.

## What this does

Given a photo of a rice leaf, the model predicts which of three conditions
it shows. The pipeline:

1. **Data preparation.** Images organised by class, split into train/val
   sets, with metadata saved as compressed CSVs to support chunked loading
   via custom Python generators, rather than loading the full image set into
   memory at once.
2. **Augmentation.** `albumentations`-based transforms (horizontal/vertical
   flips, and others) applied to the training generator to improve
   robustness on a small dataset.
3. **Model.** MobileNet, pretrained on ImageNet, with the final
   classification layer replaced by a new 3-class dense output and
   fine-tuned on the rice leaf dataset using the Adam optimiser.
4. **Evaluation.** Validation accuracy/loss, a confusion matrix, and a full
   per-class classification report (precision, recall, F1) on held-out
   validation images.

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **0.933** |
| Validation loss | 0.428 |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Bacterial leaf blight | 1.00 | 1.00 | 1.00 |
| Brown spot | 0.83 | 1.00 | 0.91 |
| Leaf smut | 1.00 | 0.80 | 0.89 |
| **Accuracy** | | | **0.93** |
| Macro avg | 0.94 | 0.93 | 0.93 |
| Weighted avg | 0.94 | 0.93 | 0.93 |

The model perfectly separates bacterial leaf blight from the other two
classes. Nearly all of its errors come from confusing brown spot and leaf
smut with each other, a visually reasonable failure mode since both present
as small discoloured leaf lesions, unlike the more distinct blight symptoms.

## Tech stack

Python, TensorFlow/Keras (`MobileNet`, transfer learning), `albumentations`
(image augmentation), `scikit-learn` (confusion matrix, classification
report), `pandas` and OpenCV (data handling and image I/O).

## Data

Uses a public rice leaf disease image dataset (three classes: bacterial
leaf blight, brown spot, leaf smut). The raw images aren't included in this
repo. Download the dataset separately and update the input paths in the
notebook before running.

## Running it

```bash
pip install tensorflow albumentations opencv-python scikit-learn pandas matplotlib
jupyter notebook main.ipynb
```
