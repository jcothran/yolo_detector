import os
from typing import Union
import cv2
import numpy as np
from ultralytics import YOLO #FIX? - if SAHI/slicing already includes this, might be able to remove this import and other code related to pulling the YOLO model with reference to the previous single pass processing?
from pathlib import Path
from score import ClassificationModelResult, BoundingBoxPoint
from model_version import ModelFramework, YOLOModelName, YOLOModelVersion
from metrics import increment_yolo_counter
import logging

#from sahi.utils.yolov8 import (
#    download_yolov8s_model,
#)

from sahi import AutoDetectionModel
#from sahi.utils.cv import read_image
#from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
#from IPython.display import Image

logger = logging.getLogger( __name__ )

font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

YOLO_MODELS = {
    "best_yolo": {
        "1": YOLO(str(MODEL_FOLDER / "yolo" / "best_yolo" / "1" / "yolov8n.pt" )),
    },
    "best_seal": {
        "1": YOLO(str(MODEL_FOLDER / "yolo" / "best_seal" / "1" / "best_seal.pt" )),
    }    
}

def yolo_process_image(
    yolo_model: YOLO,
    output_path: Path,
    model: Union[YOLOModelName, str],
    version: Union[YOLOModelVersion, str],
    yolo_threshold: float,
    api_slice_width: int,
    api_slice_height: int,
    cls_names_valid: str,
    gpu_choice: str,    
    name: str,
    bytedata: bytes
):

    assert yolo_model, \
        f"Must have yolo_model passed to {yolo_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {yolo_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {yolo_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( YOLOModelName, str ) )
    assert isinstance( version, ( YOLOModelVersion, str ) )


    if( isinstance( model, YOLOModelName ) ):
        model = model.value

    if( isinstance( version, YOLOModelVersion ) ):
        version = version.value



    ret: ClassificationModelResult = ClassificationModelResult(
        ModelFramework.YOLO.name,
        model,
        version
    )

    output_file = output_path / model / str(version) / name

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_boxes = frame

    #use YOLOv8
    #results = yolo_model.predict(frame, conf = yolo_threshold)

    #could add other .pt file choices here based on model/version settings
    pt_file = "yolov8n.pt" #default file
    
    if model == "best_seal" and version == "1":
        pt_file ="best_seal.pt"

    device_choice = "cpu" #default choice
    if gpu_choice == "Y":
        device_choice = "cuda:0"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(MODEL_FOLDER / "yolo" / model / version / pt_file ),
        confidence_threshold=yolo_threshold,
        #device="cpu",  # or 'cuda:0'
        device=device_choice,
    )

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height = api_slice_height, 
        slice_width = api_slice_width, 
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )

    # If any score is above threshold, flag it as detected
    detected = False

    # The result is a list of predictions.
    predictions = result.object_prediction_list
    print(f'There were {len(predictions)} predictions total!')

    detections_passed = 0
    
    for prediction in predictions:
        bbox = prediction.bbox

        cls_name = prediction.category.name
        
        score = prediction.score.value
        if score < yolo_threshold:
            continue

        #FIX - should be able to filter for only class id of interest in function flags
        #if cls_name == "boat" or cls_name == "horse" or cls_name == "bus" or cls_name == "train" or cls_name == "motorcycle":
        #    continue

        #if cls_name != "bird":
        #    continue
        
        if cls_name not in cls_names_valid:
            continue

        if cls_name is not None:
            print (cls_name) #a dictionary name lookup based on integer index
            
            detected = True
            detections_passed += 1
            
            x1 = int(bbox.minx)
            y1 = int(bbox.miny)
            x2 = int(bbox.maxx)
            y2 = int(bbox.maxy)
 
            # Update Prometheus metrics
            increment_yolo_counter(
                ModelFramework.YOLO.name,
                model,
                version,
                cls_name 
            )

            label = cls_name + ": " + ": {:d}%".format(int(score * 100))             
            img_boxes = cv2.rectangle(img_boxes, (x1, y2), (x2, y1), (255, 255, 255), 1)
            cv2.putText(img_boxes, label, (x1, y2 - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            ret.add(
                classification_name=cls_name,
                classification_score=score,
                bbox=(
                    BoundingBoxPoint( x1, y1 ),
                    BoundingBoxPoint( x2, y2 ),
                )
            )
        else:
            raise Exception(
                f"Classification {cls_name} not handled, model names "
                f"are: {repr(yolo_model.names)}"
            )

    if detected is True:

        print(f'There were {detections_passed} detections passed!')

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), img_boxes )
        return ( str(output_file), ret )

    return ( None, ret  )
