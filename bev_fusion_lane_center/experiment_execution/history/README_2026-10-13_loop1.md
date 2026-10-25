# ML-Experiment: Spurmitten aus BEV-Fusion (Kamera + Radar)

Ziel: Pruefen, wie weit sich aus den eingefrorenen BEV-Merkmalen eines Kamera+Radar-Netzes Spurmitten (Mittellinien) schaetzen, mit einer Karte ausrichten, der richtigen Spur zuordnen
und zeitlich verfolgen lassen. Testdaten: nuScenes (nur nicht kommerzielle Nutzung).

## Stand 2026-10-13

Fundament: **CRN** (Camera Radar Net, ICCV 2023, auf Objekterkennung trainiert, eingefroren; die Pipeline liest keine Lidar-Datei). Ein pixelbasierter Decoder sagt den Abstand zur naechsten Mittellinie vorher (200 x 200, 0,5 m),
Schwelle + Ausduennen liefern Polylinien; die Karten-Knoten (alle 3 m) werden unter synthetischer Drift ausgerichtet, den Spuren zugeordnet und mit einem Kalman-Tracker verfolgt.

- Schaetzer-F1 bei 1 m (1.706 neue Validierungs-Samples): Standard-Linienbildung 0,613, fein abgestimmt 0,670 (Netz: unet64, Schwelle 0,15 m).
- Mehr Daten helfen stark, ein groesserer Decoder deutlich (U-Netz statt FCN +0,092 F1 bei 1 m); Zeitfusion der Felder nichts; Kalman-Tracking der Drift bei grosser Drift; Ausbeute und Praezision fallen mit der Entfernung.
- Sparsifizierung (E26): mit perfektem Feld bleibt die Linienbildung bei F1 ca. 0,99, der Engpass ist das Feld; ein Punkt-fuer-Punkt-Linienverfolger ist gleichwertig zu Schwelle + Skelett und laesst sich verlustfrei auf Knoten alle 3 m ausduennen.
- Naechster Hebel: ein Kamera+Radar-Fundament, das auf Karten-Segmentierung statt Objekterkennung trainiert ist. Recherche: **GaussianCaR** (ICRA 2026, Apache-2.0, Gewichte auf Hugging Face, Spurteiler-IoU 50,1 auf nuScenes, gleiches 200 x 200-Raster); Umgebung `gcar` und Probelauf sind in Vorbereitung (`scripts/gcar_tools.py`).

| Bericht | Experiment | Kernaussage |
|---|---|---|
| R15 | Kamera-Radar-Basis (CRN) auf Mini: Layout, Merkmalsquelle und Kanalzahl | CRN-Merkmale auf Mini: Layout geprueft (Spiegelungen deutlich schlechter), Heatmap-F1 0,58; Kanalzahl und Merkmalsquelle aendern nichts |
| R16 | Kamera-Radar auf Mini: Decoder-Aenderungen und Sensorseiten vor der Fusion | Mini: groesserer Decoder und Spiegeln helfen nicht; Radar allein (0,50) traegt fast so viel wie Kamera allein (0,51) |
| R17 | Kamera+Radar: Skalierung auf Teile 1-3 (FCN, U-Netz) | Skalierung auf Teile 1-3: Daten x1,9, U-Netz deutlich besser als FCN |
| R18 | Moderner dichter Decoder: CNT-Net (ConvNeXt V2 + Transformer) auf Kamera+Radar-Merkmalen | CNT-Net (ConvNeXt V2 + Transformer, 6.000 Schritte) schlaegt das U-Netz im Rechenbudget nicht |
| R19 | Linienbildung fein abstimmen: Schwelle, Glaettung, Mindestlaenge (Kamera+Radar) | kleinere Schwelle (0,15 m) hebt den Schaetzer um ca. 0,06 F1; Glaettung schadet |
| R20 | Kausale Zeitfusion der Abstandsfelder mit Ego-Bewegung (Kamera+Radar) | kausale Zeitfusion der Felder: kein Gewinn |
| R21 | Karte unter Drift: Ausrichtung, Spurzuordnung und Route (Kamera+Radar) | Karte unter Drift: Ausrichtung, Spurzuordnung der Routenknoten ca. 80 -> 90 % |
| R22 | Zeitliches Tracking der Karten-Drift mit Kalman-Filter (Kamera+Radar) | Kalman-Tracker auf dem Drift-Zustand senkt den Fehler bei grosser Drift |
| R23 | Ensemble der Decoder-Familien (FCN, U-Netz, CNT-Net) auf Kamera+Radar-Merkmalen | Ensemble der Decoder: schlechter als der beste Einzeldecoder |
| R24 | Fehleranalyse nach Entfernung und Ort (Kamera+Radar) | Ausbeute und Praezision fallen mit der Entfernung |
| R25 | Koordinatenkanaele als Eingabe (Entfernungs-Bewusstsein) und entfernungsabhaengige Schwelle | Koordinatenkanaele: grosser Effekt beim kurz trainierten FCN (nicht mit vollem Training geprueft); entfernungsabhaengige Schwelle ohne Gewinn |
| R26 | Sparsifizierung auf dem Abstandsfeld: Punkt-fuer-Punkt-Linienverfolger mit Linienpruefung | Sparsifizierung: mit perfektem Feld F1 ca. 0,99, der Engpass ist das Feld; Linienverfolger gleichwertig zum Skelett, 3-m-Knoten ohne Verlust, mehr Schleifen in Kreuzungen |

Gesamtbericht: `reports\Gesamtbericht_Spurmitten_BEV.html` und `.pdf`.

**Lidar:** Der Lidar+Kamera-Strang (BEVFusion, E01-E14) wurde am 2026-10-13 geloescht: zuerst (Missverstaendnis) die Rohpunktwolken `LIDAR_TOP` (ca. 135 GB, Archive nicht mehr vorhanden), danach auf ausdruecklichen Wunsch alle Ergebnisse
(Merkmale ~31 GB, Ergebnis-Ordner, Berichte E01-E14, Lidar-Zahlen und -Bilder in allen Berichten); Protokoll `out\loeschprotokoll_lidar_ergebnisse.txt`. Code und BEVFusion-Gewichte (`~/work/bevfusion`) blieben erhalten.

## Ordner

- `data\nuscenes` - entpackter Mini-Datensatz mit Karten (`maps\expansion` enthaelt die Spurdaten); Teile 1-6 des vollen Datensatzes liegen im WSL-Laufwerk (`~/data/nuscenes_full`)
- `scripts` - Setup-, Auswertungs- und Berichtsskripte (siehe `reports\README.md`)
- `out` - Ergebnisse, Logs, Bilder (Kamera+Radar unter `out\crn`)
- `reports` - Einzelberichte und Gesamtbericht
- `wsl` - Linux-Laufwerk (Ubuntu 22.04, Benutzer `ml`, 24 GB RAM-Limit)

## Wie die Soll-Daten entstehen

Die Karte (`maps\expansion\*.json`) enthaelt pro Ort Spuren (`lane`) und Spurverbinder (`lane_connector`) mit Kreisbogen-Pfaden. Das Devkit tastet diese zu Mittellinien ab. `centerline_gt.py` (mit `GT_FRAME=ego`) rechnet sie in das
Fahrzeug-System der Hinterachse zum Kamera-Zeitpunkt um (x nach rechts, y nach vorn) und zeichnet sie in ein 200 x 200-Raster (+-50 m, 0,5 m Zellen); `prep_graph.py` erzeugt die Karten-Knoten alle 3 m, `make_dist_gt.py` die exakte Abstandskarte.

## Umgebungen (WSL2, Ubuntu 22.04)

- `crn` - CRN (Python 3.8, PyTorch 1.9.1 + CUDA 11.1, mmcv 1.6.0, mmdet 2.25.2; pip < 24.1, torchmetrics 0.9.3): Merkmals-Extraktion (`crn_extract.py`).
- `bev` - Auswertungen und Berichte (Python 3.8, PyTorch 1.10.1 + CUDA 11.3, numba, scipy, matplotlib); enthaelt noch BEVFusion (nicht mehr benutzt).
- `gcar` - GaussianCaR (Python 3.12, PyTorch 2.8 + CUDA 12.9, spconv, torch-scatter; Gauss-Rasterizer mit nvcc 12.9 gebaut, `scripts/setup_gcar_env.sh`, `gcar_build_rast2.sh`, `gcar_patch_rast.sh`).
- Hinweise: WSL-DNS mit eigener `resolv.conf` (`scripts\fix_dns.sh`); Anaconda-Standardquellen verlangen eine Zustimmung, daher nur conda-forge (`--override-channels`); nicht mehrere RAM-schwere Laeufe gleichzeitig (24 GB).

## Daten (nuScenes)

Mini (4 Orte, 404 Keyframes, 10 Szenen) und die Teile 1-6 des Trainval-Satzes (Teile 1-3 ausgewertet: 10.120 Samples, 8.414 Training / 1.706 Validierung). Rohdaten aus dem offiziellen Bucket `motional-nuscenes` (Registry of Open Data on AWS,
nuscenes.org verlinkt, kein Login), Pruefung ueber die S3-ETag (`scripts\download_nuscenes.ps1`; die mitgelieferte `md5.checksum` ist veraltet). Kamera-Keyframes und Radar (5 Sensoren, 8 Sweeps) werden gelesen, Lidar-Daten nicht.

## Hinweise

- nuScenes ist nur fuer nicht-kommerzielle Nutzung freigegeben; CRN steht unter MIT-Lizenz; GaussianCaR unter Apache-2.0 (sein Gauss-Rasterizer nur fuer Forschung frei).
- Geloeschte Dateien geben im WSL-Laufwerk (`ext4.vhdx`) keinen Platz an Windows zurueck; Kompaktierung mit `Optimize-VHD`/`diskpart` (Administratorrechte).
