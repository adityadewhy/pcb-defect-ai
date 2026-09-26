"""
evaluate.py

This is the REAL evaluation script. Unlike test_preprocessing.py (which just
counted boxes), this one checks predictions against the actual ground-truth
defect locations from your dataset's XML annotation files.

WHY THIS MATTERS:
Counting "how many boxes did the model draw" tells us nothing about whether
those boxes were CORRECT. A model that draws 50 random boxes everywhere
would "win" that kind of count, while being useless. This script checks each
predicted box against the real answer key (the XML files) and reports:

  PRECISION = "Of everything the model flagged, how much was actually real?"
              (High precision = few false alarms)

  RECALL    = "Of all the real defects that exist, how many did it find?"
              (High recall = few missed defects)

  F1 SCORE  = a single number that balances both (higher is better overall)

NO IMAGE RESIZING BEFORE YOLO:
We removed the img.thumbnail((640, 640)) step here. That was only ever added
to save RAM on Render's free tier (512MB limit). Running locally, we have
plenty of RAM, so we feed YOLO the full original photo. Note: YOLO still
internally resizes the image to imgsz=640 before it goes into the neural
network -- that's just how the model architecture works and can't be skipped
-- but that's YOLO doing a clean, single resize internally, instead of us
manually shrinking the image beforehand and handing YOLO an already-degraded
version.

HOW TO USE:
1. Make sure this file, preprocessing.py, and best.pt are all in your
   backend/ folder.
2. Make sure your ground-truth XML files are in the annotations/ folder,
   one level above backend/ (matching your file structure):
       pcb-defect-ai/
         annotations/        <- .xml files live here
         backend/
           test_images/      <- matching .jpg files live here
           evaluate.py        <- this file
   If your XML filenames don't exactly match your image filenames (minus
   the extension), edit the find_annotation_file() function below.
3. Run: python evaluate.py
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
from ultralytics import YOLO

from preprocessing import preprocess_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'test_images')
ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'annotations')

REAL_CONF_THRESHOLD = 0.04   # matches what you actually run in production
IOU_MATCH_THRESHOLD = 0.3    # how much overlap counts as "a correct hit"
                              # (0.5 is the common academic standard, but PCB
                              # defects are tiny, so 0.3 is a fairer bar here.
                              # Feel free to try 0.5 too and report both.)

# NMS (Non-Max Suppression) IoU thresholds to test. This controls how much
# two predicted boxes need to overlap before YOLO treats them as "the same
# defect" and throws one away. Default is 0.7 (70% overlap needed). Since
# PCB defects are tiny, two boxes can sit on the same real defect while
# overlapping much less than 70%, so both survive as "duplicates" that get
# counted as false alarms. Testing lower values checks if that's the issue.
NMS_IOU_VALUES = [0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

model = YOLO(MODEL_PATH)

CONFIGS = {
    "none": {"use_clahe": False, "use_denoise": False},
}


def find_annotation_file(image_filename):
    """Given '01_missing_hole_01.jpg', looks for '01_missing_hole_01.xml'."""
    stem = os.path.splitext(image_filename)[0]
    xml_path = os.path.join(ANNOTATIONS_DIR, stem + '.xml')
    return xml_path if os.path.exists(xml_path) else None


def parse_ground_truth(xml_path):
    """
    Reads an XML annotation file and returns a list of real defect boxes:
    [(xmin, ymin, xmax, ymax), ...]
    (We ignore the class name and just check "is there a real defect here",
    since we're evaluating detection quality, not classification accuracy.)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bnd = obj.find('bndbox')
        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def compute_iou(box_a, box_b):
    """
    Measures how much two boxes overlap, from 0.0 (no overlap) to 1.0
    (identical). This is the standard way to check "did this predicted
    box actually land on the real defect".
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold):
    """
    Compares predicted boxes to real defect boxes.
    Returns (true_positives, false_positives, false_negatives) counts for
    this single image.
    """
    gt_matched = [False] * len(gt_boxes)
    true_positives = 0
    false_positives = 0

    # Check each predicted box against all not-yet-matched real defects,
    # and take the best overlap.
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_index = -1
        for i, gt_box in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        if best_iou >= iou_threshold and best_gt_index != -1:
            gt_matched[best_gt_index] = True
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = gt_matched.count(False)
    return true_positives, false_positives, false_negatives


def get_preprocessed_image(pil_image, config_opts):
    """Applies preprocessing (if any) once, so we don't redo slow denoising repeatedly."""
    if config_opts["use_clahe"] or config_opts["use_denoise"]:
        return preprocess_image(pil_image, **config_opts)
    return pil_image


def run_inference(preprocessed_image, nms_iou):
    """Runs YOLO on an already-preprocessed image with a given NMS IoU threshold."""
    results = model.predict(
        source=preprocessed_image,
        conf=REAL_CONF_THRESHOLD,
        imgsz=640,
        iou=nms_iou,
        verbose=False
    )
    result = results[0]
    pred_boxes = [tuple(box.xyxy[0].tolist()) for box in result.boxes]
    return pred_boxes


def main():
    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    # totals[(config_name, nms_iou)] = {"tp": 0, "fp": 0, "fn": 0}
    totals = {
        (name, nms_iou): {"tp": 0, "fp": 0, "fn": 0}
        for name in CONFIGS
        for nms_iou in NMS_IOU_VALUES
    }

    skipped = []

    for filename in image_files:
        xml_path = find_annotation_file(filename)
        if xml_path is None:
            skipped.append(filename)
            continue

        gt_boxes = parse_ground_truth(xml_path)
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        # NOTE: no thumbnail/resize here -- full original resolution goes in
        original_img = Image.open(img_path).convert('RGB')

        print(f"\n{filename}  (real defects in this image: {len(gt_boxes)})")
        for name, opts in CONFIGS.items():
            # Preprocess ONCE per config (denoising is slow -- don't repeat it
            # for every NMS value we test below)
            preprocessed = get_preprocessed_image(original_img, opts)

            row = f"  {name:<16}"
            for nms_iou in NMS_IOU_VALUES:
                pred_boxes = run_inference(preprocessed, nms_iou)
                tp, fp, fn = match_predictions_to_ground_truth(pred_boxes, gt_boxes, IOU_MATCH_THRESHOLD)
                totals[(name, nms_iou)]["tp"] += tp
                totals[(name, nms_iou)]["fp"] += fp
                totals[(name, nms_iou)]["fn"] += fn
                row += f" | nms={nms_iou}: pred={len(pred_boxes)},correct={tp},false={fp}"
            print(row)

    if skipped:
        print(f"\n(Skipped {len(skipped)} images with no matching annotation file: {skipped})")

    print("\n" + "=" * 80)
    print(f"{'Config':<18}{'NMS IoU':<10}{'Precision':<14}{'Recall':<14}{'F1 Score'}")
    print("-" * 80)
    for name in CONFIGS:
        for nms_iou in NMS_IOU_VALUES:
            counts = totals[(name, nms_iou)]
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            print(f"{name:<18}{nms_iou:<10}{precision:<14.3f}{recall:<14.3f}{f1:.3f}")

    print("\nPrecision = of everything flagged, how much was a REAL defect")
    print("Recall    = of all REAL defects, how many did it actually find")
    print("F1        = balance of both (higher is better)")
    print("\nIf lower NMS IoU values show HIGHER precision with similar recall,")
    print("that confirms duplicate overlapping boxes were the problem.")


if __name__ == '__main__':
    main()
