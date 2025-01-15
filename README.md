
# Beginner's Hypothesis 2y (25)

## Weakly Supervised Object Localisation and Triplet Detection

### Challenge Background 
This project is inspired by the CholecTriplet 2022 Challenge and aims to contribute to the advancement of fine-grained surgical activity modeling in computer-assisted surgery.
For further details, refer to the official CholecTriplet GitHub repository.

Formalizing surgical activities as triplets of the used instruments, actions performed, and target anatomies acted upon provides a better understanding of surgical workflows. Automatic recognition of these triplet activities directly from surgical videos has the potential to revolutionize intra-operative decision support systems, enhancing safety and efficiency in the operating room (OR).

##
### Objective 
The challenge involves developing algorithms to:

1) Localize Instruments and Targets: Identify the spatial regions of likelihood for instruments and targets within video frames.
2) Recognize Action Triplets: Predict triplets of the form {instrument, verb, target} for each video frame.
3) Associate Triplets with Bounding Boxes: Link action triplets to their corresponding localized bounding boxes.
 

/

This is a weakly supervised learning problem where spatial annotations are not provided during training.
.


### Dataset

The dataset used for this challenge is a subset of the **CholecT50** endoscopic video dataset, which is annotated with action triplet labels.

- **Dataset structure:**

  ```bash
  ├── CholecT50
  │   ├── videos
  │   │   ├── VID01
  │   │   │   ├── 000000.png
  │   │   │   ├── ...
  │   │   │   └── N.png
  │   │   ├── ...
  │   │   └── VIDN
  │   │       ├── 000000.png
  │   │       ├── ...
  │   │       └── N.png
  │   ├── labels
  │   │   ├── VID01.json
  │   │   ├── ...
  │   │   └── VIDNN.json
  │   ├── label_mapping.txt        
  │   ├── LICENSE
  │   └── README.md



## Method 
### Part1 -Weakly Supervised Learning 
- RESNET - Used  a pre-trained Resnet-50 model for the task of tool count prediction. Replaced the final full connected layer to output 6 surgical tool classes. Used Adam Optimizer and MSE as the loss function 

- CAM - Used Grad Cam to extract most relevant areas in the input image that the resnet used to make predictions that are tool regions . Extracted Bounding Boxes using this.

### Part2 -Triplet Detection 
- Multi task temporal model - Consists of  Resnet as a feature extractor followed by 2 masked multi-head attention layers to incorporate temporal context.These features are then passed to 3 different prediction branches instrument detection, verb detection, target detection. The output logits of each of these branches are then concatenated along with the original features and passed to a triplet prediction branch.If there are no detected instruments in the frame this model is not run.
- Loss function-Used Weighted Binary Cross Entropy

### Merging 
- Used the detected tool-id by the first model to extract the corresponding detected triplet ids from the second  model.
- If there is no detected tool-id in the frame the triplet probabilities are set to zero and the triplet,instrumentid,bbox coordinates to -1
- Tool probabilities are appended using the second model











