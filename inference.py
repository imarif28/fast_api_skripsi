"""
inference.py - ONNX inference engine untuk model YOLOv8s deteksi tanda tangan
Menangani: pre-processing, inferensi, NMS, dan filtering single/multiple bbox
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from dotenv import load_dotenv
from personality import MULTIPLE_BBOX_CLASSES

load_dotenv()

# ==============================================================================
# Nama kelas sesuai urutan label YOLO (indeks 0-13)
# 3 label dihapus dari model terbaru:
#   - ujung_garis_ke_bawah
#   - ujung_garis_garis_datar
#   - ujung_garis_menghujam_bawah
# ==============================================================================
CLASS_NAMES: list[str] = [
    "awal_garis_dari_atas",       # 0
    "awal_garis_dari_bawah",      # 1
    "coretan_badan",              # 2
    "garis_bawah",                # 3
    "garis_berbalik_ke_belakang", # 4
    "huruf_pertama_besar",        # 5
    "huruf_pertama_garis_tegas",  # 6
    "huruf_pertama_lengkung_atas",# 7
    "huruf_pertama_lingkaran",    # 8
    "huruf_pertama_segitiga",     # 9
    "jarak_kosong",               # 10
    "ornamen",                    # 11
    "pertemuan_garis",            # 12
    "ujung_garis_ke_atas",        # 13
]

NUM_CLASSES = len(CLASS_NAMES)  # 14


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Konversi bounding box dari format cx,cy,w,h ke x1,y1,x2,y2."""
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return result


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Non-Maximum Suppression (NMS) manual menggunakan numpy.
    Returns daftar indeks yang lolos seleksi.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        # Hitung IoU antara box pertama dengan semua box lainnya
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class SignatureDetector:
    """
    Engine inferensi tanda tangan menggunakan model YOLOv8s (format ONNX).
    """

    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "models/best.onnx")
        self.conf_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        self.iou_threshold  = float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.input_size     = 640  # Ukuran input model

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model ONNX tidak ditemukan di: '{model_path}'. "
                "Pastikan file best.onnx sudah ada di folder models/"
            )

        # Optimalisasi CPU Multithreading untuk mempercepat inference (Standard Skripsi)
        options = ort.SessionOptions()
        options.intra_op_num_threads = os.cpu_count() or 4

        # Gunakan CPU provider untuk menghindari konflik GPU/CUDA
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] Model ONNX berhasil dimuat dari: {model_path}")

    def _preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, int, int, int, int, float]:
        """
        Pra-pemrosesan gambar:
        1. Decode bytes ke BGR array
        2. Resize ke 640x640 dengan letterbox padding
        3. Normalisasi ke [0,1]
        4. Transpose ke [1,3,H,W]

        Returns:
            (input_tensor, orig_h, orig_w, scale_x, scale_y)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Gambar tidak dapat di-decode. Pastikan format JPG/PNG yang valid.")

        orig_h, orig_w = img_bgr.shape[:2]

        # Letterbox resize: pertahankan aspek rasio
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Buat canvas hitam 640x640 dan tempel gambar di tengah
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalisasi dan transpose: HWC (BGR) -> CHW (RGB) -> NCHW
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(img_float, (2, 0, 1))[np.newaxis, ...]  # [1,3,640,640]

        # Faktor skala untuk rescale bounding box ke dimensi asli
        scale_x = (orig_w / new_w) * (new_w / self.input_size) if new_w > 0 else 1
        scale_y = (orig_h / new_h) * (new_h / self.input_size) if new_h > 0 else 1

        return input_tensor, orig_h, orig_w, pad_x, pad_y, scale

    def _postprocess(
        self,
        output: np.ndarray,
        orig_h: int,
        orig_w: int,
        pad_x: int,
        pad_y: int,
        scale: float,
    ) -> tuple[list[dict], dict[str, float]]:
        """
        Post-processing output YOLOv8:
        Output shape: [1, 4+num_classes, 8400] = [1, 18, 8400]
        
        Langkah:
        1. Transpose ke [8400, 18]
        2. Pisahkan bbox (4) dan class scores (14)
        3. Filter berdasarkan confidence
        4. Jalankan NMS per kelas
        5. Rescale bbox ke dimensi gambar asli
        6. Filter single/multiple bbox sesuai aturan

        Returns:
            Tuple (final_detections, all_confidences_dict)
        """
        # output shape: [1, 18, 8400] -> [8400, 18]
        predictions = output[0].T  # [8400, 18]

        # Pisahkan koordinat dan scores
        boxes_raw  = predictions[:, :4]              # [8400, 4] cx,cy,w,h (normalized 0-640)
        class_scores = predictions[:, 4:]            # [8400, 14]

        # Rekam confidence tertinggi untuk ke-14 kelas
        max_conf_per_class = np.max(class_scores, axis=0)
        all_confidences = {
            CLASS_NAMES[i]: round(float(max_conf_per_class[i]), 4)
            for i in range(NUM_CLASSES)
        }

        # Dapatkan confidence tertinggi dan kelas per anchor
        confidences = np.max(class_scores, axis=1)   # [8400]
        class_ids   = np.argmax(class_scores, axis=1) # [8400]

        # Filter berdasarkan confidence threshold
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return [], all_confidences

        boxes_raw   = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        # Konversi cx,cy,w,h -> x1,y1,x2,y2 (koordinat dalam skala 640x640)
        boxes_xyxy = _xywh_to_xyxy(boxes_raw)

        # NMS per kelas untuk menghilangkan duplikat tumpang-tindih
        raw_detections: dict[int, list[dict]] = {}
        for cls_id in np.unique(class_ids):
            cls_mask  = class_ids == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_confs = confidences[cls_mask]

            keep_idxs = _nms(cls_boxes, cls_confs, self.iou_threshold)

            for idx in keep_idxs:
                box = cls_boxes[idx]
                conf = float(cls_confs[idx])

                # Rescale koordinat dari canvas 640x640 ke dimensi gambar asli
                x1 = max(0.0, (box[0] - pad_x) / scale)
                y1 = max(0.0, (box[1] - pad_y) / scale)
                x2 = min(float(orig_w), (box[2] - pad_x) / scale)
                y2 = min(float(orig_h), (box[3] - pad_y) / scale)

                det = {
                    "label_id"   : int(cls_id),
                    "class_name" : CLASS_NAMES[int(cls_id)],
                    "confidence" : round(conf, 4),
                    "bounding_box": [float(round(x1, 2)), float(round(y1, 2)), float(round(x2, 2)), float(round(y2, 2))],
                }

                if int(cls_id) not in raw_detections:
                    raw_detections[int(cls_id)] = []
                raw_detections[int(cls_id)].append(det)

        # ── Aturan Single vs Multiple Bounding Box ──────────────────────────
        final_detections: list[dict] = []
        for cls_id, dets in raw_detections.items():
            class_name = CLASS_NAMES[cls_id]
            if class_name in MULTIPLE_BBOX_CLASSES:
                # Kembalikan semua deteksi untuk kelas ini
                final_detections.extend(dets)
            else:
                # Kembalikan hanya 1 deteksi dengan confidence tertinggi
                best = max(dets, key=lambda d: d["confidence"])
                final_detections.append(best)

        return final_detections, all_confidences

    def detect(self, image_bytes: bytes) -> tuple[list[dict], dict[str, float]]:
        """
        Endpoint utama: terima bytes gambar, return daftar deteksi.

        Args:
            image_bytes: Bytes dari file gambar (JPG/PNG).

        Returns:
            Tuple: (List dict deteksi yang sudah difilter, Dict all confidences)
        """
        input_tensor, orig_h, orig_w, pad_x, pad_y, scale = self._preprocess(image_bytes)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections, all_confs = self._postprocess(outputs[0], orig_h, orig_w, pad_x, pad_y, scale)
        return detections, all_confs
