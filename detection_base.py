import cv2
import torch
import time
import threading
import winsound
import csv
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ===================== CONFIGURATION GENERALE =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERAL_MODEL_PATH = "yolov8s.pt"          # modèle "small" -> plus précis que nano
CONFIDENCE_THRESHOLD = 0.35
PPE_CONFIDENCE_THRESHOLD = 0.35            # le modèle EPI a un mAP plus faible, seuil permissif
IOU_THRESHOLD = 0.45
IMG_SIZE = 832

CAM_WIDTH, CAM_HEIGHT = 1280, 720

NON_COMPLIANT_CLASSES = {"no_helmet", "no_glove", "no_goggles", "no_mask", "no_shoes"}
COMPLIANT_CLASSES = {"helmet", "glove", "goggles", "mask", "shoes"}
ALERT_COOLDOWN_SEC = 2.0

# ===================== CONFIGURATION LOGGING CHANTIER =====================
LOG_DIR = "chantier_logs"
SCREENSHOT_DIR = os.path.join(LOG_DIR, "captures")
CSV_PATH = os.path.join(LOG_DIR, "violations_log.csv")
LOG_COOLDOWN_PER_PERSON_SEC = 10.0  # ne relog/recapture la même personne que toutes les 10s

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "person_id", "equipement_manquant", "capture"])

last_logged_per_person = {}   # {tid: last_log_timestamp}
total_violations_logged = 0

print(f"[INFO] Device : {DEVICE} | Modele general : {GENERAL_MODEL_PATH} | imgsz={IMG_SIZE}")

# ===================== CHARGEMENT MODELES =====================
print("[INFO] Chargement du modele general...")
general_model = YOLO(GENERAL_MODEL_PATH)
general_model.to(DEVICE)

print("[INFO] Telechargement/chargement du modele EPI (chantier)...")
ppe_weights_path = hf_hub_download(
    repo_id="keremberke/yolov8n-protective-equipment-detection",
    filename="best.pt"
)
ppe_model = YOLO(ppe_weights_path)
ppe_model.to(DEVICE)

print("[INFO] Chargement du modele de description (BLIP)...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
print("[INFO] Tous les modeles sont prets.")

# ===================== ETAT PARTAGE (THREADS) =====================
caption_cache = {}
caption_pending = set()
lock = threading.Lock()
last_alert_time = 0


# ===================== FONCTIONS UTILITAIRES =====================
def generate_caption_async(track_id, crop_bgr, label):
    """Genere la description dans un thread separe pour ne pas bloquer la video."""
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = caption_processor(pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = caption_model.generate(**inputs, max_new_tokens=30)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        with lock:
            caption_cache[track_id] = caption
            caption_pending.discard(track_id)
        print(f"[DESCRIPTION] ID:{track_id} ({label}) -> {caption}")
    except Exception as e:
        with lock:
            caption_pending.discard(track_id)
        print(f"[ERREUR description] {e}")


def play_alert_sound():
    try:
        winsound.Beep(1000, 200)
    except Exception:
        pass


def draw_label(frame, text, x, y, color, scale=0.5, thickness=1):
    """Affiche du texte avec un fond semi-transparent pour la lisibilite."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def box_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_associated_person(epi_box, persons):
    """Trouve la personne (boite) qui contient le centre de la boite EPI. Renvoie le tid ou None."""
    cx, cy = box_center(*epi_box)
    best_tid = None
    best_area = float("inf")
    for tid, (px1, py1, px2, py2) in persons.items():
        margin_x = int((px2 - px1) * 0.15)
        margin_y = int((py2 - py1) * 0.15)
        if (px1 - margin_x) <= cx <= (px2 + margin_x) and (py1 - margin_y) <= cy <= (py2 + margin_y):
            area = (px2 - px1) * (py2 - py1)
            if area < best_area:
                best_area = area
                best_tid = tid
    return best_tid


def log_violation(frame, tid, missing_set):
    """Enregistre une violation EPI : CSV + capture d'ecran, avec cooldown par personne."""
    global total_violations_logged
    now = time.time()
    last_log = last_logged_per_person.get(tid, 0)

    if now - last_log < LOG_COOLDOWN_PER_PERSON_SEC:
        return

    last_logged_per_person[tid] = now
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_name = f"violation_ID{tid}_{timestamp_str}.jpg"
    screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)

    cv2.imwrite(screenshot_path, frame)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tid,
            ", ".join(sorted(missing_set)),
            screenshot_name
        ])

    total_violations_logged += 1
    print(f"[LOG] Violation enregistree -> ID:{tid} | Manque: {', '.join(missing_set)} | {screenshot_path}")


# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam")
    exit()

DESCRIBE_MODE = False
CHANTIER_MODE = False
fps_history = []

print("[CONTROLES] 'c' = mode Chantier | 'd' = decrire objets | 'm' = afficher descriptions | 'q' = quitter")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        CHANTIER_MODE = not CHANTIER_MODE
        print(f"[INFO] Mode CHANTIER {'ACTIVE' if CHANTIER_MODE else 'DESACTIVE'}")
    if key == ord('m'):
        DESCRIBE_MODE = not DESCRIBE_MODE
        print(f"[INFO] Affichage description {'ACTIVE' if DESCRIBE_MODE else 'DESACTIVE'}")

    current_ids_visible = set()
    alert_triggered_this_frame = False

    if not CHANTIER_MODE:
        # ===================== MODE GENERAL =====================
        results = general_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False,
            conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
            device=DEVICE, half=(DEVICE == "cuda")
        )[0]

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = general_model.names[cls_id]
                tid = int(track_id)
                current_ids_visible.add(tid)

                color = (0, 255, 0) if label == "person" else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"ID:{tid} {label} {conf:.2f}", x1, y1 - 6, color)

                if key == ord('d'):
                    with lock:
                        already_running = tid in caption_pending
                    if not already_running:
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            with lock:
                                caption_pending.add(tid)
                            threading.Thread(target=generate_caption_async,
                                              args=(tid, crop, label), daemon=True).start()

                if DESCRIBE_MODE:
                    with lock:
                        caption = caption_cache.get(tid)
                    if caption:
                        draw_label(frame, caption, x1, y2 + 20, (255, 255, 255), scale=0.45)
                    elif tid in caption_pending:
                        draw_label(frame, "...", x1, y2 + 20, (200, 200, 200), scale=0.45)

    else:
        # ===================== MODE CHANTIER (EPI) =====================
        # 1) Detection + tracking des personnes (modele general, classe "person" uniquement)
        person_results = general_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False,
            conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
            device=DEVICE, half=(DEVICE == "cuda"), classes=[0]  # 0 = "person" dans COCO
        )[0]

        persons = {}  # tid -> (x1,y1,x2,y2)
        if person_results.boxes.id is not None:
            for box, track_id in zip(person_results.boxes, person_results.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                tid = int(track_id)
                persons[tid] = (x1, y1, x2, y2)
                current_ids_visible.add(tid)

        # 2) Detection des equipements (modele EPI)
        ppe_results = ppe_model.predict(
            frame, verbose=False, conf=PPE_CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD, imgsz=IMG_SIZE, device=DEVICE
        )[0]

        # 3) Association equipement <-> personne
        missing_items = {tid: set() for tid in persons}
        present_items = {tid: set() for tid in persons}

        for box in ppe_results.boxes:
            ex1, ey1, ex2, ey2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = ppe_model.names[cls_id]

            tid = find_associated_person((ex1, ey1, ex2, ey2), persons)
            if tid is None:
                continue

            if label in NON_COMPLIANT_CLASSES:
                missing_items[tid].add(label.replace("no_", ""))
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 255), 1)
            elif label in COMPLIANT_CLASSES:
                present_items[tid].add(label)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 200, 0), 1)

        # 4) Affichage + alerte + log par personne
        for tid, (x1, y1, x2, y2) in persons.items():
            missing = missing_items.get(tid, set())
            if missing:
                color = (0, 0, 255)
                status = f"ID:{tid} MANQUE: {', '.join(sorted(missing))}"
                alert_triggered_this_frame = True
                log_violation(frame.copy(), tid, missing)
            else:
                color = (0, 255, 0)
                status = f"ID:{tid} CONFORME"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, status, x1, y1 - 6, color, scale=0.5, thickness=2)

    # --- Alerte securite globale ---
    if CHANTIER_MODE and alert_triggered_this_frame:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        draw_label(frame, "ALERTE: EQUIPEMENT DE SECURITE MANQUANT",
                   20, h - 20, (0, 0, 255), scale=0.7, thickness=2)

        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN_SEC:
            last_alert_time = now
            threading.Thread(target=play_alert_sound, daemon=True).start()
            print("[ALERTE SECURITE] Equipement de protection manquant detecte !")

    # Nettoyage descriptions des objets disparus
    with lock:
        for tid in list(caption_cache.keys()):
            if tid not in current_ids_visible:
                del caption_cache[tid]

    # FPS lisse (moyenne glissante sur 15 frames)
    curr_time = time.time()
    instant_fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    fps_history.append(instant_fps)
    if len(fps_history) > 15:
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)

    mode_label = "CHANTIER (EPI)" if CHANTIER_MODE else "GENERAL"
    desc_status = "DESC: ON" if DESCRIBE_MODE else "DESC: OFF"
    draw_label(frame, f"FPS: {avg_fps:.1f} | MODE: {mode_label} | {desc_status} | Personnes/Objets: {len(current_ids_visible)}",
               10, 30, (0, 255, 255), scale=0.55, thickness=2)

    cv2.imshow("Detecteur Universel - General / Chantier", frame)

    if key == ord('q'):
        break

print(f"\n[RESUME SESSION] Total de violations enregistrees : {total_violations_logged}")
print(f"[RESUME SESSION] Logs disponibles dans : {os.path.abspath(CSV_PATH)}")
print(f"[RESUME SESSION] Captures disponibles dans : {os.path.abspath(SCREENSHOT_DIR)}")

cap.release()
cv2.destroyAllWindows()