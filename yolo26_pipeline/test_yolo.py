## this page of code is made in order to test the accuracy of the yolo model that has already been created in comparison to human counted images of grapes

from pathlib import Path
import csv

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "wgisd_seg" / "images" / "test"
RUNS_DIR = ROOT / "runs"

data = [
    ("SYH_2017-04-27_1322.jpg", 20),
    ("SYH_2017-04-27_1318.jpg", 8),
    ("SYH_2017-04-27_1304.jpg", 7),
    ("SYH_2017-04-27_1269.jpg", 11),
    ("SYH_2017-04-27_1239.jpg", 21),
    ("SVB_20180427_151818928.jpg", 21),
    ("SVB_1972.jpg", 22),
    ("SVB_1954.jpg", 19),
    ("SVB_1944.jpg", 18),
    ("SVB_1935.jpg", 25),
    ("CSV_20180427_144723166_HDR.jpg", 14),
    ("CSV_20180427_144535647.jpg", 13),
    ("CSV_20180427_144507419.jpg", 16),
    ("CSV_1898.jpg", 11),
    ("CSV_1877.jpg", 8),
    ("CFR_1666.jpg", 12),
    ("CFR_1651.jpg", 14),
    ("CFR_1643.jpg", 15),
    ("CFR_1641.jpg", 30),
    ("CFR_1639.jpg", 17),
    ("CFR_1638.jpg", 10),
    ("CDY_2054.jpg", 16),
    ("CDY_2051.jpg", 11),
    ("CDY_2046.jpg", 10),
    ("CDY_2043.jpg", 10),
    ("CDY_2040.jpg", 8),
    ("CDY_2015.jpg", 20),
]


def main():
    # Find the most recently modified trained best.pt
    checkpoints = list(
        RUNS_DIR.glob("yolo11_wgisd*/weights/best.pt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No trained best.pt found in {RUNS_DIR}. "
            "Check your training run directory."
        )

    model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {model_path}")

    model = YOLO(str(model_path))

    output_csv = ROOT / "yolo_test_results.csv"
    (ROOT / "test_predictions").mkdir(exist_ok=True)
    results_data = []

    for filename, actual_count in data:
        image_path = IMAGE_DIR / filename

        if not image_path.is_file():
            print(f"Missing image: {image_path}")
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=512,conf=0.25,verbose=False,)[0]

 # Count detected instance masks.
 # For segmentation, boxes and masks correspond to instances.
        predicted_count = (
        len(result.masks) if result.masks is not None else 0
        )

        error = predicted_count - actual_count

        results_data.append({
            "image": filename,"actual_count": actual_count,"predicted_count": predicted_count,
            "error": error,"absolute_error": abs(error),})

        print(f"{filename:38} "f"Actual: {actual_count:2} | "f"Predicted: {predicted_count:2} | "f"Error: {error:+3}")

        # Save visual predictions for inspection.
        result.save(filename=str(ROOT / "test_predictions" / filename))

    if not results_data:
        print("No images were evaluated.")
        return

    mae = sum(r["absolute_error"] for r in results_data) / len(results_data)
    exact = sum(
        r["actual_count"] == r["predicted_count"]
        for r in results_data
 )

    total_actual = sum(r["actual_count"] for r in results_data)
    total_predicted = sum(r["predicted_count"] for r in results_data)

    print("\n--- Evaluation summary ---")
    print(f"Images evaluated: {len(results_data)}/{len(data)}")
    print(f"Mean absolute count error: {mae:.2f} objects/image")
    print(f"Exact count matches: {exact}/{len(results_data)} "f"({100 * exact / len(results_data):.1f}%)")
    print(f"Total actual objects: {total_actual}")
    print(f"Total predicted objects: {total_predicted}")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nCSV saved to: {output_csv}")
    print(f"Annotated images saved to: {ROOT / 'test_predictions'}")


if __name__ == "__main__":
        main()