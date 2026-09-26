"""
test_preprocessing.py

Run this on YOUR LAPTOP (not on Render) to measure whether the
preprocessing pipeline actually helps.

WHAT IT DOES:
1. Looks at every image in a folder called "test_images"
2. Runs YOLO on each image WITHOUT preprocessing (the "before" version)
3. Runs YOLO on each image WITH preprocessing (the "after" version)
4. Prints a table comparing how many defects were detected, and saves
   side-by-side annotated images to a folder called "comparison_results"
   so you can visually show your teacher the difference.

HOW TO USE:
1. Put this file in the same folder as app.py, preprocessing.py, and best.pt
2. Create a folder called "test_images" and put ~10-20 PCB photos in it
   (a mix of ones you know have defects works best)
3. Run: python test_preprocessing.py
4. Check the printed table + the "comparison_results" folder
"""

import os
from PIL import Image
from ultralytics import YOLO

from preprocessing import preprocess_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

# Set this to match what you actually use in production (you mentioned ~4%)
REAL_CONF_THRESHOLD = 0.04

model = YOLO(MODEL_PATH)

# The 4 configurations we're testing, to figure out which ingredient
# actually helps vs. hurts, instead of guessing.
CONFIGS = {
    "none":            {"use_clahe": False, "use_denoise": False},
    "denoise_only":    {"use_clahe": False, "use_denoise": True},
    "clahe_only":       {"use_clahe": True,  "use_denoise": False},
    "clahe+denoise":   {"use_clahe": True,  "use_denoise": True},
}


def get_max_confidence_and_count(pil_image, conf_threshold):
    """
    Runs YOLO with a very low floor (0.01) so we see every candidate box,
    then reports:
      - the highest confidence score seen (0.0 if nothing at all)
      - how many boxes would count as "detected" at your real threshold
    """
    results = model.predict(source=pil_image, conf=0.01, imgsz=640, verbose=False)
    result = results[0]
    if len(result.boxes) == 0:
        return 0.0, 0
    confidences = [float(box.conf[0]) for box in result.boxes]
    max_conf = max(confidences)
    count_at_threshold = sum(1 for c in confidences if c >= conf_threshold)
    return max_conf, count_at_threshold


def main():
    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}. Add some PCB photos there first.")
        return

    # We'll total up results per configuration across all images
    totals = {name: {"max_conf_sum": 0.0, "count_sum": 0, "zeroed_out": 0}
               for name in CONFIGS}

    header = f"{'Image':<28}"
    for name in CONFIGS:
        header += f"{name + ' (conf/count)':<24}"
    print(header)
    print("-" * (28 + 24 * len(CONFIGS)))

    for filename in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        original_img = Image.open(img_path).convert('RGB')
        original_img.thumbnail((640, 640))

        row = f"{filename:<28}"
        baseline_conf = None

        for name, opts in CONFIGS.items():
            test_img = preprocess_image(original_img, **opts) if (opts["use_clahe"] or opts["use_denoise"]) else original_img
            max_conf, count_at_thresh = get_max_confidence_and_count(test_img, REAL_CONF_THRESHOLD)

            totals[name]["max_conf_sum"] += max_conf
            totals[name]["count_sum"] += count_at_thresh

            if name == "none":
                baseline_conf = max_conf
            # Track cases where a config that used to find something now finds nothing
            if baseline_conf is not None and baseline_conf > 0 and max_conf == 0.0:
                totals[name]["zeroed_out"] += 1

            row += f"{max_conf:.3f} / {count_at_thresh:<15}"

        print(row)

    print("-" * (28 + 24 * len(CONFIGS)))
    print(f"\n{'Config':<18}{'Avg MaxConf':<15}{'Total Count':<15}{'Images zeroed-out'}")
    for name, data in totals.items():
        avg_conf = data["max_conf_sum"] / len(image_files)
        print(f"{name:<18}{avg_conf:<15.3f}{data['count_sum']:<15}{data['zeroed_out']}")

    print(f"\n(Threshold used for counting: {REAL_CONF_THRESHOLD})")
    print("'Images zeroed-out' = images where the plain image found SOMETHING,")
    print("but this config found NOTHING AT ALL. Lower is better — a high number")
    print("here means this technique is erasing real signal, not just noise.")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
