for bh 24-25
Beginner's Hypothesis 2025 (2Y)
Surgical Action Triplet Post Challenge Phase
Challenge Background
This project is inspired by the CholecTriplet 2022 Challenge and aims to contribute to the advancement of fine-grained surgical activity modeling in computer-assisted surgery.
For further details, refer to the official CholecTriplet GitHub repository.

Formalizing surgical activities as triplets of the used instruments, actions performed, and target anatomies acted upon provides a better understanding of surgical workflows. Automatic recognition of these triplet activities directly from surgical videos has the potential to revolutionize intra-operative decision support systems, enhancing safety and efficiency in the operating room (OR).

Objective
The challenge involves developing algorithms to:

Localize Instruments and Targets: Identify the spatial regions of likelihood for instruments and targets within video frames.
Recognize Action Triplets: Predict triplets of the form {instrument, verb, target} for each video frame.
Associate Triplets with Bounding Boxes: Link action triplets to their corresponding localized bounding boxes.
This is a weakly supervised learning problem where spatial annotations are not provided during training.

Dataset
The dataset used for this challenge is a subset of the CholecT50 endoscopic video dataset, which is annotated with action triplet labels.

Dataset structure:
|──CholecT50
  ├───videos
  │   ├───VID01
  │   │   ├───000000.png
  │   │   ├───
  │   │   └───N.png
  │   ├───
  │   └───VIDN
  │       ├───000000.png
  │       ├───
  │       └───N.png
  ├───labels
  │   ├───VID01.json
  │   ├───
  │   └───VIDNN.json
  ├───label_mapping.txt        
  ├───LICENSE
  └───README.md


