from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# MODEL SETTINGS
DEFAULT_MODEL = (
    "hf://killuminati1/"
    "construction-ppe-yolov8/"
    "best.pt"
)

WORKER_CONFIDENCE = 0.15
HELMET_CONFIDENCE = 0.40
NO_HELMET_CONFIDENCE = 0.30

# Internal YOLO settings
DEFAULT_IOU = 0.20
DEFAULT_IMAGE_SIZE = 1280



WORKER_CLASSES = {
    "worker",
    "person",
}

HELMET_CLASSES = {
    "hard_hat",
    "hardhat",
    "helmet",
    "helmet_on",
}

NO_HELMET_CLASSES = {
    "no-helmet",
    "no_helmet",
    "no-hardhat",
    "no_hardhat",
    "helmet_off",
}


# MODEL PATH
def resolve_model_path(model_path: str) -> str:
    """
    Resolve either:

        Local model:
            C:/models/best.pt

        Hugging Face model:
            hf://username/repository/file.pt
    """

    if not model_path:
        raise ValueError(
            "YOLO model path is empty."
        )

    # Hugging Face
    if model_path.startswith("hf://"):

        try:
            from huggingface_hub import hf_hub_download

        except ImportError:

            raise ImportError(
                "huggingface_hub is not installed.\n"
                "Run:\n"
                "pip install huggingface_hub"
            )

        value = model_path[len("hf://"):]

        parts = value.split("/", 2)

        if len(parts) != 3:

            raise ValueError(
                "Invalid Hugging Face model path.\n"
                "Expected:\n"
                "hf://username/repository/file.pt"
            )

        username = parts[0]
        repository = parts[1]
        filename = parts[2]

        print(
            f"Loading PPE model: "
            f"{username}/{repository}/{filename}"
        )

        downloaded_path = hf_hub_download(
            repo_id=(
                f"{username}/{repository}"
            ),
            filename=filename,
        )

        return downloaded_path

    # Local model
    path = Path(model_path)

    if not path.exists():

        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    return str(path)


# CLASS NORMALIZATION
def normalize_class_name(
    name: str
) -> str:

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def classify_class(
    name: str
) -> str:
    """
    Converts the model class into only one of:

        worker
        helmet
        no_helmet
        ignore
    """

    name = normalize_class_name(name)

    if name in WORKER_CLASSES:

        return "worker"

    if name in HELMET_CLASSES:

        return "helmet"

    if name in NO_HELMET_CLASSES:

        return "no_helmet"

    return "ignore"


# BOX HELPERS
def box_center(
    box: List[int]
) -> Tuple[float, float]:

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def box_width(
    box: List[int]
) -> float:

    return max(
        1.0,
        box[2] - box[0]
    )


def box_height(
    box: List[int]
) -> float:

    return max(
        1.0,
        box[3] - box[1]
    )


def calculate_iou(
    box1: List[int],
    box2: List[int]
) -> float:
    """
    Calculate IoU between two boxes.
    """

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        / union_area
    )


# WORKER DEDUPLICATION
def remove_duplicate_workers(
    workers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate Worker detections while preserving
    nearby/partially overlapping workers.

    We intentionally use a moderate IoU threshold because
    construction images can contain workers close together.
    """

    if not workers:

        return []

    # Highest confidence first
    workers = sorted(
        workers,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    kept_workers = []

    for worker in workers:

        duplicate = False

        for existing in kept_workers:

            iou = calculate_iou(
                worker["box"],
                existing["box"],
            )

            # Only remove if boxes heavily overlap.
            if iou >= 0.65:

                duplicate = True
                break

        if not duplicate:

            kept_workers.append(
                worker
            )

    return kept_workers


# HELMET → WORKER MATCH SCORE
def helmet_match_score(
    helmet_box: List[int],
    worker_box: List[int],
) -> float:
    """
    Determine how likely a helmet belongs to a worker.

    This is deliberately flexible because workers can be:

        - standing
        - bending
        - leaning
        - turned sideways
        - carrying material
        - partially occluded

    Returns:
        >= 0  valid match
        < 0   invalid match
    """

    wx1, wy1, wx2, wy2 = worker_box

    hx1, hy1, hx2, hy2 = helmet_box

    worker_w = box_width(
        worker_box
    )

    worker_h = box_height(
        worker_box
    )

    helmet_w = box_width(
        helmet_box
    )

    helmet_h = box_height(
        helmet_box
    )

    helmet_cx, helmet_cy = box_center(
        helmet_box
    )

    worker_cx, worker_cy = box_center(
        worker_box
    )

    # 1. HEAD REGION
    head_region_bottom = (
        wy1
        + worker_h * 0.45
    )

    if helmet_cy > head_region_bottom:

        return -1.0

    # 2. EXPANDED HORIZONTAL WORKER AREA
    expanded_x1 = (
        wx1
        - worker_w * 0.25
    )

    expanded_x2 = (
        wx2
        + worker_w * 0.25
    )

    if helmet_cx < expanded_x1:

        return -1.0

    if helmet_cx > expanded_x2:

        return -1.0

    # 3. HORIZONTAL OVERLAP
    overlap_width = max(
        0,
        min(
            hx2,
            wx2
        )
        -
        max(
            hx1,
            wx1
        )
    )

    overlap_ratio = (
        overlap_width
        / helmet_w
    )

    # Some side-view workers have small overlap.
    if overlap_ratio < 0.10:

        return -1.0

    # 4. HORIZONTAL CENTER DISTANCE
    horizontal_distance = abs(
        helmet_cx
        - worker_cx
    )

    normalized_horizontal_distance = (
        horizontal_distance
        / worker_w
    )

    if (
        normalized_horizontal_distance
        > 0.75
    ):

        return -1.0

    # 5. HELMET SIZE
    helmet_width_ratio = (
        helmet_w
        / worker_w
    )

    if helmet_width_ratio < 0.025:

        return -1.0

    # Extremely large detections are suspicious.
    if helmet_width_ratio > 0.75:

        return -1.0

    # 6. HELMET HEIGHT
    helmet_height_ratio = (
        helmet_h
        / worker_h
    )

    if helmet_height_ratio > 0.25:

        return -1.0

    # 7. VERTICAL SCORE
    vertical_distance = (
        helmet_cy
        - wy1
    ) / worker_h

    vertical_score = max(
        0.0,
        1.0
        -
        vertical_distance * 2.2
    )

    # 8. HORIZONTAL SCORE
    horizontal_score = max(
        0.0,
        1.0
        -
        normalized_horizontal_distance
    )

    # 9. OVERLAP SCORE
    overlap_score = min(
        1.0,
        overlap_ratio
    )

    # 10. SIZE SCORE
    size_score = 1.0

    if helmet_width_ratio > 0.55:

        size_score = 0.65

    elif helmet_width_ratio < 0.05:

        size_score = 0.75

    # FINAL SCORE
    score = (
        vertical_score * 0.35
        +
        horizontal_score * 0.30
        +
        overlap_score * 0.25
        +
        size_score * 0.10
    )

    return float(score)


# FIND BEST HELMET / NO-HELMET FOR WORKER
def find_best_status_for_worker(
    worker: Dict[str, Any],
    ppe_detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Find the most appropriate helmet status for a worker.

    Important:
        Worker detection is completely independent.

    Even if no helmet is found, the worker remains counted.
    """

    helmet_candidates = []

    no_helmet_candidates = []

    for detection in ppe_detections:

        score = helmet_match_score(
            detection["box"],
            worker["box"],
        )

        if score < 0:

            continue

        candidate = (
            score,
            detection["confidence"],
            detection,
        )

        if (
            detection["category"]
            == "helmet"
        ):

            helmet_candidates.append(
                candidate
            )

        elif (
            detection["category"]
            == "no_helmet"
        ):

            no_helmet_candidates.append(
                candidate
            )

    # SORT
    helmet_candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    no_helmet_candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    # NO HELMET
    if no_helmet_candidates:

        score, confidence, detection = (
            no_helmet_candidates[0]
        )

        if score >= 0.25:

            return {
                "status": "Without Helmet",

                "ppe_confidence": confidence,

                "ppe_class": detection[
                    "class"
                ],

                "match_score": round(
                    score,
                    3,
                ),
            }

    # HELMET
    if helmet_candidates:

        score, confidence, detection = (
            helmet_candidates[0]
        )

        # Helmet requires both confidence and spatial match.
        if (
            confidence >= HELMET_CONFIDENCE
            and
            score >= 0.30
        ):

            return {
                "status": "With Helmet",

                "ppe_confidence": confidence,

                "ppe_class": detection[
                    "class"
                ],

                "match_score": round(
                    score,
                    3,
                ),
            }

    # UNKNOWN
    return {
        "status": "Unknown",

        "ppe_confidence": None,

        "ppe_class": None,

        "match_score": None,
    }


# DRAW WORKER BOX
def draw_worker_box(
    image: np.ndarray,
    worker: Dict[str, Any],
) -> None:
    """
    Draw ONLY the worker bounding box.

    No helmet boxes.
    No PPE boxes.
    """

    x1, y1, x2, y2 = worker["box"]

    status = worker["status"]

    worker_id = worker[
        "worker_id"
    ]

    # LABEL
    if status == "With Helmet":

        label = (
            f"Worker {worker_id} - Helmet"
        )

        # Green
        color = (
            0,
            180,
            0,
        )

    elif status == "Without Helmet":

        label = (
            f"Worker {worker_id} - No Helmet"
        )

        # Red
        color = (
            0,
            0,
            255,
        )

    else:

        label = (
            f"Worker {worker_id}"
        )

        # Yellow
        color = (
            0,
            200,
            255,
        )

    # WORKER BOX
    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        3,
    )

    # TEXT
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.60

    thickness = 2

    (
        text_size,
        baseline,
    ) = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    text_width = text_size[0]

    text_height = text_size[1]

    label_top = max(
        0,
        y1
        -
        text_height
        -
        baseline
        -
        6,
    )

    # LABEL BACKGROUND
    cv2.rectangle(
        image,
        (
            x1,
            label_top,
        ),
        (
            x1
            +
            text_width
            +
            10,
            y1,
        ),
        color,
        -1,
    )

    # LABEL TEXT
    cv2.putText(
        image,
        label,
        (
            x1 + 5,
            y1 - 6,
        ),
        font,
        font_scale,
        (
            255,
            255,
            255,
        ),
        thickness,
        cv2.LINE_AA,
    )


# MAIN ANALYZER
def analyze_labour(
    image: np.ndarray,
    model_path: str = DEFAULT_MODEL,
    confidence: float = WORKER_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Any]:
    """
    Main construction worker + helmet analyzer.

    Returns:

        workers
        with_helmet
        without_helmet
        helmet_compliance
        worker_status
        annotated_image
    """

    # VALIDATE IMAGE
    if image is None:

        raise ValueError(
            "Input image is None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise TypeError(
            "image must be a NumPy array."
        )

    if image.size == 0:

        raise ValueError(
            "Input image is empty."
        )

    # LOAD MODEL
    resolved_model = resolve_model_path(
        model_path
    )

    print(
        f"Using YOLO model: "
        f"{resolved_model}"
    )

    model = YOLO(
        resolved_model
    )

    # YOLO DETECTION
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou_threshold,
        imgsz=image_size,
        verbose=False,
    )

    # DETECTION STORAGE
    workers = []

    ppe_detections = []

    # READ MODEL OUTPUT
    for result in results:

        if result.boxes is None:

            continue

        boxes = result.boxes

        for i in range(
            len(boxes)
        ):

            # CLASS
            cls_id = int(
                boxes.cls[i].item()
            )

            # CONFIDENCE
            confidence_value = float(
                boxes.conf[i].item()
            )

            # BOX
            xyxy = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy,
            )

            # CLASS NAME
            class_name = model.names.get(
                cls_id,
                str(cls_id),
            )

            # CLASSIFY
            category = classify_class(
                class_name
            )

            # IGNORE ALL OTHER PPE
            if category == "ignore":

                continue

            detection = {
                "class": class_name,

                "category": category,

                "confidence": round(
                    confidence_value,
                    3,
                ),

                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }

            # WORKER
            if category == "worker":

                # Worker confidence is controlled by YOLO's
                # main confidence parameter.
                workers.append(
                    detection
                )

            # HELMET
            elif category == "helmet":

                if (
                    confidence_value
                    >= HELMET_CONFIDENCE
                ):

                    ppe_detections.append(
                        detection
                    )

            # NO HELMET
            elif category == "no_helmet":

                if (
                    confidence_value
                    >= NO_HELMET_CONFIDENCE
                ):

                    ppe_detections.append(
                        detection
                    )

    # WORKER DEDUPLICATION
    workers = remove_duplicate_workers(
        workers
    )

    # SORT WORKERS LEFT → RIGHT
    workers.sort(
        key=lambda worker: (
            worker["box"][0]
            +
            worker["box"][2]
        ) / 2
    )

    # ASSIGN STATUS
    worker_status = []
    with_helmet = 0
    without_helmet = 0

    for worker_id, worker in enumerate(
        workers,
        start=1,
    ):

        status_result = (
            find_best_status_for_worker(
                worker,
                ppe_detections,
            )
        )

        status = status_result[
            "status"
        ]

        # COUNTS
        if status == "With Helmet":

            with_helmet += 1

        elif status == "Without Helmet":

            without_helmet += 1

        # STORE WORKER
        worker_result = {
            "worker_id": worker_id,

            "box": worker["box"],

            "confidence": worker[
                "confidence"
            ],

            "status": status,

            "ppe_confidence": (
                status_result[
                    "ppe_confidence"
                ]
            ),

            "ppe_class": (
                status_result[
                    "ppe_class"
                ]
            ),

            "match_score": (
                status_result[
                    "match_score"
                ]
            ),
        }

        worker_status.append(
            worker_result
        )

    # TOTAL WORKERS
    total_workers = len(
        worker_status
    )

    # HELMET COMPLIANCE
    if total_workers > 0:

        helmet_compliance = (
            with_helmet
            /
            total_workers
        ) * 100

    else:

        helmet_compliance = 0.0

    # ANNOTATED IMAGE
    annotated_image = image.copy()

    for worker in worker_status:

        draw_worker_box(
            annotated_image,
            worker,
        )

    # SUMMARY
    summary = {
        "workers": total_workers,

        "with_helmet": with_helmet,

        "without_helmet": without_helmet,

        "helmet_compliance": round(
            helmet_compliance,
            1,
        ),
    }

    # RETURN
    return {
        "summary": summary,

        "workers": total_workers,

        "worker_status": worker_status,

        "annotated_image": annotated_image,

        # Only relevant PPE detections are returned internally.
        "detections": ppe_detections,
    }


# COMPATIBILITY ALIAS
analyze_image = analyze_labour


# OPTIONAL COMMAND-LINE TEST
if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python labour_analyzer.py image.jpg"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise SystemExit(
            f"Could not read image: {image_path}"
        )

    result = analyze_labour(
        image=image,
        model_path=DEFAULT_MODEL,
    )

    print(
        "\n================================"
    )

    print(
        "LABOUR ANALYSIS"
    )

    print(
        "================================"
    )

    print(
        f"Workers: "
        f"{result['summary']['workers']}"
    )

    print(
        f"With Helmet: "
        f"{result['summary']['with_helmet']}"
    )

    print(
        f"Without Helmet: "
        f"{result['summary']['without_helmet']}"
    )

    print(
        f"Helmet Compliance: "
        f"{result['summary']['helmet_compliance']}%"
    )

    print(
        "================================"
    )

    output_path = (
        Path(image_path).stem
        + "_annotated.jpg"
    )

    cv2.imwrite(
        output_path,
        result["annotated_image"],
    )

    print(
        f"Saved: {output_path}"
    )