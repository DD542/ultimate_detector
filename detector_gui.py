import cv2
import torch
import time
import threading
import queue
import winsound
import csv
import os
from datetime import datetime

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ===================== CONFIGURATION GENERALE =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERAL_MODEL_PATH = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.35
PPE_CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 832

CAM_WIDTH, CAM_HEIGHT = 1280, 720
DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540   # taille d'affichage dans la fenetre

NON_COMPLIANT_CLASSES = {"no_helmet", "no_glove", "no_goggles", "no_mask", "no_shoes"}
COMPLIANT_CLASSES = {"helmet", "glove", "goggles", "mask", "shoes"}
ALERT_COOLDOWN_SEC = 2.0

# ===================== CONFIGURATION LOGGING CHANTIER =====================
LOG_DIR = "chantier_logs"
SCREENSHOT_DIR = os.path.join(LOG_DIR, "captures")
CSV_PATH = os.path.join(LOG_DIR, "violations_log.csv")
LOG_COOLDOWN_PER_PERSON_SEC = 10.0

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "person_id", "equipement_manquant", "capture"])

last_logged_per_person = {}
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

# ===================== ETAT PARTAGE (THREAD VIDEO <-> GUI) =====================
state_lock = threading.Lock()
app_state = {
    "mode": "general",          # "general" ou "chantier"
    "describe_requested": False,
    "describe_mode": True,
    "running": True,
    "fps": 0.0,
    "object_count": 0,
    "alert_active": False,
    "last_violations": [],      # lignes de texte pour l'affichage dans le journal GUI
}

frame_queue = queue.Queue(maxsize=1)

caption_cache = {}
caption_pending = set()
cap_lock = threading.Lock()
last_alert_time = 0


# ===================== FONCTIONS UTILITAIRES =====================
def generate_caption_async(track_id, crop_bgr, label):
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = caption_processor(pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = caption_model.generate(**inputs, max_new_tokens=30)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        with cap_lock:
            caption_cache[track_id] = caption
            caption_pending.discard(track_id)
        print(f"[DESCRIPTION] ID:{track_id} ({label}) -> {caption}")
    except Exception as e:
        with cap_lock:
            caption_pending.discard(track_id)
        print(f"[ERREUR description] {e}")


def play_alert_sound():
    try:
        winsound.Beep(1000, 200)
    except Exception:
        pass


def draw_label(frame, text, x, y, color, scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def box_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_associated_person(epi_box, persons):
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
    line = f"{datetime.now().strftime('%H:%M:%S')}  ID:{tid}  Manque: {', '.join(sorted(missing_set))}"
    with state_lock:
        app_state["last_violations"].append(line)
    print(f"[LOG] {line} -> {screenshot_path}")


# ===================== BOUCLE VIDEO (THREAD SEPARE) =====================
def video_worker():
    global last_alert_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        with state_lock:
            app_state["running"] = False
        return

    fps_history = []
    prev_time = time.time()

    while True:
        with state_lock:
            running = app_state["running"]
        if not running:
            break

        ret, frame = cap.read()
        if not ret:
            break

        with state_lock:
            mode = app_state["mode"]
            describe_requested = app_state["describe_requested"]
            app_state["describe_requested"] = False
            describe_mode = app_state["describe_mode"]

        current_ids_visible = set()
        alert_triggered_this_frame = False

        if mode == "general":
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

                    if describe_requested:
                        with cap_lock:
                            already_running = tid in caption_pending
                        if not already_running:
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size > 0:
                                with cap_lock:
                                    caption_pending.add(tid)
                                threading.Thread(target=generate_caption_async,
                                                  args=(tid, crop, label), daemon=True).start()

                    if describe_mode:
                        with cap_lock:
                            caption = caption_cache.get(tid)
                        if caption:
                            draw_label(frame, caption, x1, y2 + 20, (255, 255, 255), scale=0.45)
                        elif tid in caption_pending:
                            draw_label(frame, "...", x1, y2 + 20, (200, 200, 200), scale=0.45)

        else:
            # ===================== MODE CHANTIER (EPI) =====================
            person_results = general_model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
                device=DEVICE, half=(DEVICE == "cuda"), classes=[0]
            )[0]

            persons = {}
            if person_results.boxes.id is not None:
                for box, track_id in zip(person_results.boxes, person_results.boxes.id):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    tid = int(track_id)
                    persons[tid] = (x1, y1, x2, y2)
                    current_ids_visible.add(tid)

            ppe_results = ppe_model.predict(
                frame, verbose=False, conf=PPE_CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD, imgsz=IMG_SIZE, device=DEVICE
            )[0]

            missing_items = {tid: set() for tid in persons}

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
                    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 200, 0), 1)

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

        # --- Alerte securite (overlay video) ---
        if mode == "chantier" and alert_triggered_this_frame:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            draw_label(frame, "ALERTE: EQUIPEMENT DE SECURITE MANQUANT",
                       20, h - 20, (0, 0, 255), scale=0.7, thickness=2)

            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN_SEC:
                last_alert_time = now
                threading.Thread(target=play_alert_sound, daemon=True).start()

        with cap_lock:
            for tid in list(caption_cache.keys()):
                if tid not in current_ids_visible:
                    del caption_cache[tid]

        # FPS lisse
        curr_time = time.time()
        instant_fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        fps_history.append(instant_fps)
        if len(fps_history) > 15:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        with state_lock:
            app_state["fps"] = avg_fps
            app_state["object_count"] = len(current_ids_visible)
            app_state["alert_active"] = alert_triggered_this_frame

        # Envoi de la frame vers l'interface graphique
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(rgb_frame)

    cap.release()


# ===================== INTERFACE GRAPHIQUE (TKINTER) =====================
class DetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecteur Universel - General / Chantier")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        mono_font = tkfont.Font(family="Consolas", size=10)
        title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        # --- Flux video ---
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # --- Barre de statut ---
        self.status_var = tk.StringVar(value="FPS: -- | Mode: GENERAL | Objets: 0")
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                      fg="#ffff66", bg="#1e1e1e", font=mono_font, anchor="w")
        self.status_label.grid(row=1, column=0, columnspan=4, sticky="w", padx=10)

        # --- Boutons de controle ---
        self.mode_btn = tk.Button(self.root, text="Mode : GENERAL  (cliquer -> Chantier)",
                                   command=self.toggle_mode, width=32, bg="#2d2d2d", fg="white")
        self.mode_btn.grid(row=2, column=0, padx=10, pady=8)

        self.describe_btn = tk.Button(self.root, text="Decrire les objets visibles",
                                       command=self.request_describe, width=26, bg="#2d2d2d", fg="white")
        self.describe_btn.grid(row=2, column=1, padx=10, pady=8)

        self.toggle_desc_btn = tk.Button(self.root, text="Afficher / Masquer descriptions",
                                          command=self.toggle_describe_display, width=28, bg="#2d2d2d", fg="white")
        self.toggle_desc_btn.grid(row=2, column=2, padx=10, pady=8)

        self.quit_btn = tk.Button(self.root, text="Quitter", command=self.quit_app,
                                   width=12, bg="#a83232", fg="white")
        self.quit_btn.grid(row=2, column=3, padx=10, pady=8)

        # --- Journal des violations ---
        tk.Label(self.root, text="Journal des violations EPI (mode Chantier) :",
                 fg="white", bg="#1e1e1e", font=title_font).grid(
            row=3, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 0))

        self.log_box = tk.Listbox(self.root, height=6, width=112, bg="#111111",
                                   fg="#ff7777", font=mono_font, borderwidth=0, highlightthickness=0)
        self.log_box.grid(row=4, column=0, columnspan=4, padx=10, pady=(2, 10))

        self.update_gui()

    def toggle_mode(self):
        with state_lock:
            app_state["mode"] = "chantier" if app_state["mode"] == "general" else "general"
            new_mode = app_state["mode"]
        if new_mode == "chantier":
            self.mode_btn.config(text="Mode : CHANTIER  (cliquer -> General)", bg="#7a4b00")
        else:
            self.mode_btn.config(text="Mode : GENERAL  (cliquer -> Chantier)", bg="#2d2d2d")

    def request_describe(self):
        with state_lock:
            app_state["describe_requested"] = True

    def toggle_describe_display(self):
        with state_lock:
            app_state["describe_mode"] = not app_state["describe_mode"]

    def quit_app(self):
        with state_lock:
            app_state["running"] = False
        self.root.after(300, self.root.destroy)

    def update_gui(self):
        try:
            rgb_frame = frame_queue.get_nowait()
            img = Image.fromarray(rgb_frame).resize((DISPLAY_WIDTH, DISPLAY_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk   # garde une reference (sinon l'image disparait)
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass

        with state_lock:
            fps = app_state["fps"]
            mode = app_state["mode"]
            count = app_state["object_count"]
            alert_active = app_state["alert_active"]
            violations = list(app_state["last_violations"])
            running = app_state["running"]

        self.status_var.set(f"FPS: {fps:.1f} | Mode: {mode.upper()} | Personnes/Objets: {count}")
        self.status_label.config(fg="#ff4444" if alert_active else "#ffff66")

        current_size = self.log_box.size()
        if len(violations) > current_size:
            for line in violations[current_size:]:
                self.log_box.insert(tk.END, line)
            self.log_box.yview_moveto(1.0)

        if running:
            self.root.after(20, self.update_gui)
        else:
            self.root.after(300, self.root.destroy)


def main():
    worker = threading.Thread(target=video_worker, daemon=True)
    worker.start()

    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()

    with state_lock:
        app_state["running"] = False

    print(f"\n[RESUME SESSION] Total de violations enregistrees : {total_violations_logged}")
    print(f"[RESUME SESSION] Logs disponibles dans : {os.path.abspath(CSV_PATH)}")
    print(f"[RESUME SESSION] Captures disponibles dans : {os.path.abspath(SCREENSHOT_DIR)}")


if __name__ == "__main__":
    main()
