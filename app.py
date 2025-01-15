import streamlit as st
import os
import json
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from part2.model import MultiBranchModel
import logging  # For debugging
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load triplet mapping from file
TRIPLET_FILE = "triplet.txt"
def load_triplet_mapping(file_path):
    triplet_mapping = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                triplet_id, triplet_label = line.strip().split(":")
                triplet_mapping[int(triplet_id)] = triplet_label
        st.write("Triplet mapping loaded successfully.")
    except Exception as e:
        st.write(f"Error loading triplet mapping: {e}")
    return triplet_mapping

triplet_mapping = load_triplet_mapping(TRIPLET_FILE)

# Define colors for bounding boxes
COLORS = [
    (255, 255, 255),  # White
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255)       # Red
]

# Mapping data: each entry represents [triplet_id, tool_id, ...additional mapping]
mapping_data = [
    [0, 0, 2, 1, 2, 1],
    [1, 0, 2, 0, 2, 0],
    [2, 0, 2, 10, 2, 10],
    
    [3, 0, 0, 3, 0, 3],
    [4, 0, 0, 2, 0, 2],
    [5, 0, 0, 4, 0, 4],
    [6, 0, 0, 1, 0, 1],
    [7, 0, 0, 0, 0, 0],
    [8, 0, 0, 12, 0, 12],
    [9, 0, 0, 8, 0, 8],
    [10, 0, 0, 10, 0, 10],
    [11, 0, 0, 11, 0, 11],
    [12, 0, 0, 13, 0, 13],
    [13, 0, 8, 0, 8, 0],
    [14, 0, 1, 2, 1, 2],
    [15, 0, 1, 4, 1, 4],
    [16, 0, 1, 1, 1, 1],
    [17, 0, 1, 0, 1, 0],
    [18, 0, 1, 12, 1, 12],
    [19, 0, 1, 8, 1, 8],
    [20, 0, 1, 10, 1, 10],
    [21, 0, 1, 11, 1, 11],
    [22, 1, 3, 7, 13, 22],
    [23, 1, 3, 5, 13, 20],
    [24, 1, 3, 3, 13, 18],
    [25, 1, 3, 2, 13, 17],
    [26, 1, 3, 4, 13, 19],
    [27, 1, 3, 1, 13, 16],
    [28, 1, 3, 0, 13, 15],
    [29, 1, 3, 8, 13, 23],
    [30, 1, 3, 10, 13, 25],
    [31, 1, 3, 11, 13, 26],
    [32, 1, 2, 9, 12, 24],
    [33, 1, 2, 3, 12, 18],
    [34, 1, 2, 2, 12, 17],
    [35, 1, 2, 1, 12, 16],
    [36, 1, 2, 0, 12, 15],
    [37, 1, 2, 10, 12, 25],
    [38, 1, 0, 1, 10, 16],
    [39, 1, 0, 8, 10, 23],
    [40, 1, 0, 13, 10, 28],
    [41, 1, 1, 2, 11, 17],
    [42, 1, 1, 4, 11, 19],
    [43, 1, 1, 0, 11, 15],
    [44, 1, 1, 8, 11, 23],
    [45, 1, 1, 10, 11, 25],
    [46, 2, 3, 5, 23, 35],
    [47, 2, 3, 3, 23, 33],
    [48, 2, 3, 2, 23, 32],
    [49, 2, 3, 4, 23, 34],
    [50, 2, 3, 1, 23, 31],
    [51, 2, 3, 0, 23, 30],
    [52, 2, 3, 8, 23, 38],
    [53, 2, 3, 10, 23, 40],
    [54, 2, 5, 5, 25, 35],
    [55, 2, 5, 11, 25, 41],
    [56, 2, 2, 5, 22, 35],
    [57, 2, 2, 3, 22, 33],
    [58, 2, 2, 2, 22, 32],
    [59, 2, 2, 1, 22, 31],
    [60, 2, 2, 0, 22, 30],
    [61, 2, 2, 10, 22, 40],
    [62, 2, 2, 11, 22, 41],
    [63, 2, 1, 0, 21, 30],
    [64, 2, 1, 8, 21, 38],
    [65, 3, 3, 10, 33, 55],
    [66, 3, 5, 9, 35, 54],
    [67, 3, 5, 5, 35, 50],
    [68, 3, 5, 3, 35, 48],
    [69, 3, 5, 2, 35, 47],
    [70, 3, 5, 1, 35, 46],
    [71, 3, 5, 8, 35, 53],
    [72, 3, 5, 10, 35, 55],
    [73, 3, 5, 11, 35, 56],
    [74, 3, 2, 1, 32, 46],
    [75, 3, 2, 0, 32, 45],
    [76, 3, 2, 10, 32, 55],
    [77, 4, 4, 5, 44, 65],
    [78, 4, 4, 3, 44, 63],
    [79, 4, 4, 2, 44, 62],
    [80, 4, 4, 4, 44, 64],
    [81, 4, 4, 1, 44, 61],
    [82, 5, 6, 6, 56, 81],
    [83, 5, 2, 2, 52, 77],
    [84, 5, 2, 4, 52, 79],
    [85, 5, 2, 1, 52, 76],
    [86, 5, 2, 0, 52, 75],
    [87, 5, 2, 10, 52, 85],
    [88, 5, 7, 7, 57, 82],
    [89, 5, 7, 4, 57, 79],
    [90, 5, 7, 8, 57, 83],
    [91, 5, 1, 0, 51, 75],
    [92, 5, 1, 8, 51, 83],
    [93, 5, 1, 10, 51, 85],
    [94, 0, 9, 14, 9, 14],
    [95, 1, 9, 14, 19, 29],
    [96, 2, 9, 14, 29, 44],
    [97, 3, 9, 14, 39, 59],
    [98, 4, 9, 14, 49, 74],
    [99, 5, 9, 14, 59, 89],


]
      # Truncated for brevity

# Function to map tool_id to triplets
def get_triplets_for_tool(tool_id):
    return [entry[0] for entry in mapping_data if entry[1] == tool_id]

@st.cache_resource
def load_models():
    try:
        # Tool detection model
        tool_model = models.resnet50(pretrained=False)
        tool_model.fc = torch.nn.Linear(tool_model.fc.in_features, 6)
        tool_model.load_state_dict(torch.load("trained_model_fold_1.pth", map_location=torch.device('cpu')))
        tool_model.eval()
        logging.info("Tool model loaded successfully.")

        # Triplet prediction model
        triplet_model = MultiBranchModel(100, 6, 10, 15, 5)
        triplet_model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # Load weights
        triplet_model.eval()
        logging.info("Triplet model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise e

    return tool_model, triplet_model

tool_model, triplet_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tool_model = tool_model.to(device)
triplet_model = triplet_model.to(device)

def process_image(image):
    try:
         # Convert numpy.ndarray to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        
        

        # Create sequence by repeating the single image
        seq_len = 5  # Match the model's expected sequence length
        image_sequence = image_tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        image_sequence = image_sequence.to(device)
        

        # Predict tools
        with torch.no_grad():
            tool_predictions = tool_model(image_tensor.to(device)).cpu().numpy().squeeze()
        

        # Initialize CAM extractor
        cam_extractor = GradCAM(model=tool_model, target_layers=[tool_model.layer4[-1]])

        # Process each tool prediction
        bounding_boxes = []
        tool_ids = []

        for tool_id in range(6):  # Assuming 6 possible tools
            predicted_count = max(0, int(round(tool_predictions[tool_id])))
            

            if predicted_count > 0:
                # Generate Grad-CAM heatmap
                grayscale_cam = cam_extractor(
                    input_tensor=image_tensor.to(device),
                    targets=[ClassifierOutputTarget(tool_id)]
                )[0]

                # Normalize CAM
                cam_min = grayscale_cam.min()
                cam_max = grayscale_cam.max()
                if cam_max > cam_min:
                    grayscale_cam = (grayscale_cam - cam_min) / (cam_max - cam_min)

                # Create binary mask
                threshold = 0.5
                binary_mask = (grayscale_cam >= threshold).astype(np.uint8) * 255

                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours by area and take top N based on predicted count
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:predicted_count]

                # Get bounding boxes for the contours
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
                    tool_ids.append(tool_id)

        # Predict triplets using the sequence model
        triplet_predictions = []
        with torch.no_grad():
            instrument_logits, verb_logits, target_logits, triplet_logits = triplet_model(image_sequence)
            middle_frame_idx = seq_len // 2
            triplet_probs = triplet_logits[:, middle_frame_idx].squeeze(0).cpu().numpy()
            

        # Match bounding boxes with triplets
        for bbox, tool_id in zip(bounding_boxes, tool_ids):
            possible_triplets = get_triplets_for_tool(tool_id)

            triplet_matches = [
                (triplet_id, triplet_probs[triplet_id])
                for triplet_id in possible_triplets
                if triplet_probs[triplet_id] > 0.5
            ]

            if triplet_matches:
                triplet_matches.sort(key=lambda x: x[1], reverse=True)
                top_triplet_id = triplet_matches[0][0]
                triplet_predictions.append((bbox, triplet_mapping[top_triplet_id]))
                

        return bounding_boxes, triplet_predictions

    except Exception as e:
        st.write(f"Error processing image: {e}")
        return [], []

# Streamlit UI
st.title("Instrument and Triplet Detection in Surgical Images")
st.write("Upload an image to detect instruments and their corresponding triplets.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bounding_boxes, triplet_predictions = process_image(image_rgb)

    # Overlay bounding boxes and triplets on the image
    orig_h, orig_w = image_rgb.shape[:2]
    resized_h, resized_w = 256, 448

    # Scaling factors
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    for bbox, triplet in triplet_predictions:
        x, y, w, h = bbox
        # Scale bounding box to original image size
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Overlay bounding box and triplet
        color = COLORS[hash(triplet) % len(COLORS)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, triplet, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(image, channels="BGR", caption="Detected Instruments and Triplets")


