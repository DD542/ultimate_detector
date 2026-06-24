# Ultimate Detector

Systeme de detection d'objets et de personnes en mouvement, avec suivi (tracking),
description automatique, et un mode "Chantier" qui verifie le port des equipements
de protection individuelle (EPI) : casque, gants, lunettes, masque, chaussures.

## Fonctionnalites

- Detection d'objets et de personnes en temps reel (YOLOv8)
- Suivi avec ID stable (ByteTrack) meme en mouvement
- Description automatique d'un objet/personne a la demande (BLIP)
- Mode Chantier : detection des EPI + association precise par personne
- Alerte visuelle et sonore en cas d'equipement manquant
- Journal des violations : fichier CSV + capture d'ecran horodatee
- Interface graphique (Tkinter) avec boutons cliquables

## Installation

```bash
pip install -r requirements.txt
```

Sous Windows, `winsound` est inclus nativement avec Python (aucune installation requise).

## Utilisation

Version avec interface graphique (recommandee) :
```bash
python detector_gui.py
```

Version ligne de commande (controle au clavier) :
```bash
python detection_base.py
```

### Controles (version clavier)
| Touche | Action |
|---|---|
| `c` | Bascule Mode General / Mode Chantier |
| `d` | Decrit les objets visibles |
| `m` | Affiche/masque les descriptions |
| `q` | Quitter |

## Structure du projet

```
ultimate_detector/
├── detection_base.py      # Version ligne de commande (clavier)
├── detector_gui.py        # Version avec interface graphique
├── requirements.txt
└── chantier_logs/         # Genere automatiquement (logs + captures, non versionne)
    ├── violations_log.csv
    └── captures/
```

## Modeles utilises

- Detection generale : YOLOv8s (Ultralytics)
- Detection EPI : keremberke/yolov8n-protective-equipment-detection (Hugging Face)
- Description d'image : Salesforce/blip-image-captioning-base (Hugging Face)

Tous les modeles se telechargent automatiquement au premier lancement.
