from flask import Flask, request, send_file
from flask_cors import CORS
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from torchvision import transforms
import numpy as np
import easyocr
import pandas as pd
from tqdm.auto import tqdm
import openpyxl

app = Flask(__name__)
CORS(app)

model = None
structure_model = None
device = None

def initialize_models():
    global model, structure_model, device
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)

# Call the initialization function
initialize_models()

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def objects_to_crops(img, tokens, objects, class_thresholds, padding=50):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
        cropped_img = img.crop(bbox)
        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens
        table_crops.append(cropped_table)

    return table_crops

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])
    return cell_coordinates

def apply_ocr(cropped_table, cell_coordinates):
    reader = easyocr.Reader(['en'])
    data = {}
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = reader.readtext(cell_image)
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + [""] * (max_num_columns - len(row_data))
        data[row] = row_data

    return data

def save_as_xlsx(data):
    output_xlsx = os.path.join("output", "output.xlsx")
    
    # Create a workbook and add a worksheet
    workbook = openpyxl.Workbook()
    worksheet = workbook.active

    # Write data to the worksheet
    for row, row_text in data.items():
        worksheet.append(row_text)
    
    # Save the workbook to a file
    workbook.save(output_xlsx)

@app.route('/upload', methods=['POST'])
def upload_file():
    global model, structure_model, device
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        img = Image.open(file.stream).convert("RGB")

        # Processing the image
        width, height = img.size
        resized_img = img.resize((int(0.6 * width), int(0.6 * height)))

        detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pixel_values = detection_transform(img).unsqueeze(0)
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"

        objects = outputs_to_objects(outputs, img.size, id2label)

        tokens = []
        detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }

        tables_crops = objects_to_crops(img, tokens, objects, detection_class_thresholds, padding=50)

        if not tables_crops:
            return 'No tables detected', 400

        cropped_table = tables_crops[0]['image'].convert("RGB")

        structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pixel_values = structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = structure_model(pixel_values)

        structure_id2label = structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
        cell_coordinates = get_cell_coordinates_by_row(cells)

        # Pass cropped_table and cell_coordinates to apply_ocr function
        data = apply_ocr(cropped_table, cell_coordinates)
          
        # Save as XLSX
        save_as_xlsx(data)

        return send_file(os.path.join("output", "output.xlsx"), as_attachment=True)

 if __name__=='main':
    app.run(debug=True)

