"""
ONNXRuntime-based RT-DETR detector with permissive licensing.

This module provides a thin wrapper to run RT-DETR ONNX inference and return
results in a format compatible with supervision's Detections and ByteTrack.

Model Attribution:
    RT-DETR model from PaddleDetection (Apache 2.0 License)
    Paper: "DETRs Beat YOLOs on Real-time Object Detection"
    Authors: Wenyu Lv, Shangliang Xu, Yian Zhao, et al.
    Source: https://github.com/PaddlePaddle/PaddleDetection
    arXiv: https://arxiv.org/abs/2304.08069

License Stack:
    - This code: MIT License
    - RT-DETR model: Apache 2.0 License
    - ONNXRuntime: MIT License
"""

from typing import Tuple, Dict, Any
import numpy as np
import onnxruntime as ort
import cv2
import supervision as sv


def _cxcywh_to_xyxy(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    # boxes expected in normalized cx,cy,w,h
    cx = boxes[:, 0] * img_w
    cy = boxes[:, 1] * img_h
    w = boxes[:, 2] * img_w
    h = boxes[:, 3] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


class OnnxRTDetrDetector:
    """Runs RT-DETR ONNX (PaddleDetection format) and returns sv.Detections.

    The PaddleDetection RT-DETR ONNX has 3 inputs:
    - image: [batch, 3, H, W] - the actual image
    - im_shape: [batch, 2] - original image shape (H, W)
    - scale_factor: [batch, 2] - scale factors (H_scale, W_scale)
    
    Outputs:
    - boxes/scores/classes (format varies)
    """

    def __init__(self, onnx_path: str, confidence_threshold: float = 0.4) -> None:
        # Use Intel GPU (DirectML) with CPU fallback
        providers = [
            'DmlExecutionProvider',
            'CPUExecutionProvider'
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.confidence_threshold = confidence_threshold
        
        # Get input names
        self.input_names = {inp.name: idx for idx, inp in enumerate(self.session.get_inputs())}
        
        # Find the image input tensor
        for inp in self.session.get_inputs():
            if len(inp.shape) == 4:  # [batch, C, H, W]
                self.image_input_name = inp.name
                # Get target size (ignore batch and channel dimensions)
                shape = inp.shape
                if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.target_h, self.target_w = int(shape[2]), int(shape[3])
                else:
                    self.target_h, self.target_w = 640, 640
                break
        else:
            # Fallback defaults
            self.image_input_name = self.session.get_inputs()[0].name
            self.target_h, self.target_w = 640, 640

    def _preprocess(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and normalize
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # Calculate scale factors
        scale_h = h / self.target_h
        scale_w = w / self.target_w
        
        # Prepare model inputs
        inputs = {}
        
        if self.image_input_name in self.input_names:
            inputs[self.image_input_name] = img
        if 'im_shape' in self.input_names:
            inputs['im_shape'] = np.array([[h, w]], dtype=np.float32)
        if 'scale_factor' in self.input_names:
            inputs['scale_factor'] = np.array([[scale_h, scale_w]], dtype=np.float32)
        
        return inputs, (w, h)

    def _postprocess_logits_boxes(
        self,
        pred_logits: np.ndarray,
        pred_boxes: np.ndarray,
        orig_size: Tuple[int, int],
    ) -> sv.Detections:
        logits = pred_logits[0]
        boxes = pred_boxes[0]
        probs = 1 / (1 + np.exp(-logits))  # sigmoid for multi-label; use max over classes
        class_ids = np.argmax(probs, axis=1)
        scores = probs[np.arange(probs.shape[0]), class_ids]
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=np.int64)
            )
        boxes_xyxy = _cxcywh_to_xyxy(boxes[mask], orig_size[0], orig_size[1]).astype(np.float32)
        det = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=scores[mask].astype(np.float32),
            class_id=class_ids[mask].astype(np.int64),
        )
        return det

    def _postprocess_paddledet(
        self,
        detections: np.ndarray,
        num_detections: np.ndarray,
        orig_size: Tuple[int, int],
    ) -> sv.Detections:
        """Process PaddleDetection RT-DETR output format.
        
        Args:
            detections: [N, 6] array where each row is [class_id, score, x1, y1, x2, y2]
            num_detections: [1] array with number of detections
            orig_size: (width, height) of original image
        """
        # Return empty detections with all required fields
        if detections.size == 0 or int(num_detections[0]) == 0:
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=np.int64)
            )
        
        # Get number of valid detections
        n_dets = int(num_detections[0])
        detections = detections[:n_dets]
        
        # Parse detections: [class_id, score, x1, y1, x2, y2]
        class_ids = detections[:, 0].astype(np.int64)
        scores = detections[:, 1].astype(np.float32)
        boxes = detections[:, 2:6].astype(np.float32)  # [x1, y1, x2, y2]
        
        # Scale boxes from resized space (640×640) to original image space
        # PaddleDetection RT-DETR returns boxes in the resized coordinate space
        orig_w, orig_h = orig_size
        scale_w = orig_w / self.target_w
        scale_h = orig_h / self.target_h
        boxes[:, 0] *= scale_w  # x1
        boxes[:, 1] *= scale_h  # y1
        boxes[:, 2] *= scale_w  # x2
        boxes[:, 3] *= scale_h  # y2
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=np.int64)
            )
        
        return sv.Detections(
            xyxy=boxes[mask],
            confidence=scores[mask],
            class_id=class_ids[mask],
        )
    
    def _postprocess_direct(
        self,
        outputs: Dict[str, Any],
    ) -> sv.Detections:
        boxes = None
        scores = None
        class_ids = None
        for k in outputs.keys():
            lk = k.lower()
            if boxes is None and ("boxes" in lk or "bbox" in lk):
                boxes = outputs[k]
            elif scores is None and ("scores" in lk or "score" in lk or "conf" in lk):
                scores = outputs[k]
            elif class_ids is None and ("labels" in lk or "class" in lk or "cls" in lk):
                class_ids = outputs[k]
        if boxes is None:
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=np.int64)
            )
        boxes = boxes.astype(np.float32)
        if scores is None:
            scores = np.ones((boxes.shape[0],), dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros((boxes.shape[0],), dtype=np.int64)
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                class_id=np.zeros(0, dtype=np.int64)
            )
        return sv.Detections(
            xyxy=boxes[mask],
            confidence=scores[mask].astype(np.float32),
            class_id=class_ids[mask].astype(np.int64),
        )

    def infer(self, image_bgr: np.ndarray) -> sv.Detections:
        inputs, (orig_w, orig_h) = self._preprocess(image_bgr)
        outputs = self.session.run(None, inputs)
        output_names = [o.name for o in self.session.get_outputs()]
        out_map = {name: outputs[i] for i, name in enumerate(output_names)}

        # PaddleDetection RT-DETR format: 
        # Output 0: [N, 6] - each row is [class_id, score, x1, y1, x2, y2]
        # Output 1: [1] - number of detections
        if len(outputs) >= 2:
            return self._postprocess_paddledet(outputs[0], outputs[1], (orig_w, orig_h))
        
        # Path 1: logits + boxes (cxcywh) - legacy format
        if any("pred_logits" in k for k in out_map.keys()) and any("pred_boxes" in k for k in out_map.keys()):
            return self._postprocess_logits_boxes(
                out_map[[k for k in out_map.keys() if "pred_logits" in k][0]],
                out_map[[k for k in out_map.keys() if "pred_boxes" in k][0]],
                (orig_w, orig_h),
            )

        # Path 2: direct boxes/scores/classes (xyxy) - legacy format
        return self._postprocess_direct(out_map)


