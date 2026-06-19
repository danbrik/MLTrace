## Detaillierter Ablauf einer Trainingsepoche

Dieser Abschnitt beschreibt nicht nur den groben Trainingsfluss, sondern auch genauer,
wie die Daten fuer eine Epoche ausgewaehlt, geladen, vorverarbeitet, zu Batches gebaut
und auf die GPU uebertragen werden.

### 1. Vor dem Training: Dataset wird vorbereitet

Bevor die erste Epoche startet, wird ein `MeltDataset` erzeugt. Dabei werden noch nicht
alle Bilddaten in den RAM geladen. Stattdessen wird zuerst nur festgelegt, welche Dateien
ueberhaupt zum Training gehoeren.

Der Ablauf ist:

1. Die YAML-Konfiguration (`config_path`, z. B. `data/paths.yaml`) wird geladen.
2. Daraus wird der Basisordner fuer den gewaehlten Tank bestimmt.
3. Je nach `folder_structure` werden die TIFF-Dateien gesucht.
   - Bei `flat`: alle `*.tiff` direkt im Zielordner
   - Bei `hierarchical`: Dateien in Tages-Unterordnern
4. Fuer jede Datei wird der Zeitstempel aus dem Dateinamen gelesen.
5. Es bleiben nur Bilder zwischen `start_date` und `end_date`.
6. Falls `exclude_timeranges` gesetzt sind, werden diese Zeitraeume zusaetzlich entfernt.
7. Danach wird `sampling_rate` angewendet.
   - `sampling_rate = 1`: jedes Bild bleibt erhalten
   - `sampling_rate = 2`: jedes zweite Bild
   - `sampling_rate = 10`: jedes zehnte Bild

Das Ergebnis dieses Schritts ist keine Liste geladener Bilder, sondern eine Liste
gueltiger Bildpfade.

### 2. DataLoader: Wie die Daten pro Epoche ausgewaehlt werden

Aus dem Dataset wird danach ein `DataLoader` erstellt.

Im aktuellen Training ist konfiguriert:

- `batch_size = 32`
- `shuffle = True`
- `num_workers = 16`
- `pin_memory = True`

Das bedeutet:

- Pro Trainingsschritt wird ein Batch aus 32 Bildern verwendet.
- Zu Beginn jeder Epoche wird die Reihenfolge der Samples zufaellig gemischt.
- Es laufen 16 Worker-Prozesse parallel, die Daten nachladen und vorbereiten.
- Die vorbereiteten CPU-Batches werden in pinned memory abgelegt, damit der Transfer
  zur GPU schneller erfolgen kann.

### 3. Was `shuffle=True` konkret bedeutet

Das Dataset besitzt intern eine Liste mit Bildpfaden. Diese Liste hat Indizes wie:

- Bild 0
- Bild 1
- Bild 2
- Bild 3
- ...

Mit `shuffle=True` wird zu Beginn jeder Epoche eine zufaellige Reihenfolge dieser Indizes
gebildet, zum Beispiel:

- 841
- 12
- 5500
- 77
- 204
- ...

Diese gemischte Indexreihenfolge bestimmt, welche Samples in welcher Reihenfolge
waehrend dieser Epoche geladen werden.

Wichtig:
Die fachliche Auswahl der Bilder passiert vorher ueber Tank, Zeitfilter,
Ausschlussintervalle und Sampling. `shuffle=True` aendert nicht, welche Bilder
im Dataset sind, sondern nur ihre Reihenfolge innerhalb der Epoche.

### 4. Was die 16 Worker genau machen

Die 16 Worker sind reine Lade- und Preprocessing-Prozesse.
Sie trainieren nicht das Modell und fuehren keine Gradientenberechnung aus.

Ihre Aufgabe ist:

1. Vom DataLoader einen oder mehrere Indizes erhalten
2. Fuer jeden Index das passende Sample aus dem Dataset holen
3. Das Bild von der Festplatte laden
4. Die gesamte Vorverarbeitung auf dieses einzelne Bild anwenden
5. Den fertigen Tensor an den DataLoader zurueckgeben

Die Worker laden also Samples parallel im Hintergrund, waehrend der Hauptprozess
den aktuellen Batch bereits auf der GPU verarbeitet.

### 5. Wie ein einzelnes Bild geladen wird

Sobald ein Worker einen Index bekommt, ruft er im Dataset `__getitem__(idx)` auf.

Dort passiert fuer genau ein Bild:

1. Der Dateipfad `image_paths[idx]` wird aus der vorbereiteten Pfadliste geholt.
2. Das TIFF-Bild wird von der Festplatte geoeffnet.
3. Falls aktiviert, wird es von 16-bit in 8-bit umgewandelt.
4. Falls aktiviert, wird eine zeitabhaengige Kamera-Kalibrierung angewendet.
5. Danach folgt die Torch-Preprocessing-Pipeline.
6. Optional folgen weitere Bildtransformationen wie Perspektivkorrektur.
7. Das Bild wird in `float32` umgewandelt und auf `[0,1]` skaliert.
8. Optional werden CLAHE und/oder Min-Max-Normalisierung angewendet.

Am Ende liefert `__getitem__` genau einen fertigen Tensor fuer dieses Bild zurueck.

### 6. Detaillierte Preprocessing-Reihenfolge pro Bild

Die Vorverarbeitung innerhalb von `__getitem__` laeuft in dieser Reihenfolge:

1. TIFF laden
2. Optional: 16-bit zu 8-bit konvertieren
3. Optional: zeitabhaengige Kamera-Kalibrierung
4. Torch-Preprocessing-Pipeline
   - optional Brightness-Augmentation
   - optional Rotations-Augmentation
   - optional Center-Crop
   - Resize, falls keine Perspektivtransformation aktiv ist
5. Optional: 4-Punkt-Perspektivtransformation
6. Falls Perspektivtransformation aktiv war: Resize auf `resize_to`
7. Konvertierung zu `float32`
8. Skalierung auf Wertebereich `[0,1]`
   - bei `uint8` durch Division durch `255`
   - bei `uint16` durch Division durch `65535`
9. Optional: CLAHE
10. Optional: Min-Max-Normalisierung

Wichtig:
Dieses Preprocessing passiert online waehrend des Trainings und nicht als
separater Offline-Schritt vorab.

### 7. Wie aus einzelnen Samples ein Trainingsbatch entsteht

Sobald genug Worker einzelne Bilder geladen und preprocessiert haben,
setzt der DataLoader daraus einen Batch zusammen.

Bei `batch_size = 32` werden 32 Bild-Tensoren entlang der Batch-Dimension
gestapelt.

Beispiel:

- einzelnes Bild: `1 x 840 x 840`
- fertiger Batch: `32 x 1 x 840 x 840`

Dieser Batch liegt zu diesem Zeitpunkt noch im CPU-Speicher.

### 8. Wann der Batch auf die GPU geladen wird

Die Bilder werden nicht schon im Dataset auf die GPU verschoben.

Stattdessen passiert der GPU-Transfer erst im eigentlichen Trainingsloop,
also kurz bevor das Modell den Batch verarbeitet.

Der Ablauf ist:

1. Der DataLoader liefert einen fertigen CPU-Batch
2. Dieser Batch wird mit `data.to(device, non_blocking=True)` auf die GPU kopiert
3. Erst dann startet der Forward-Pass des Autoencoders

Das bedeutet:

- Laden und Preprocessing: CPU-Seite
- Training des Modells: GPU-Seite

### 9. Was innerhalb eines Trainingsschritts passiert

Fuer jeden Batch einer Epoche laeuft dann der normale Trainingsschritt:

1. Batch von CPU auf GPU kopieren
2. Gradienten des Optimizers zuruecksetzen
3. Forward-Pass durch den Autoencoder
4. Rekonstruktion mit dem Eingabebild vergleichen
5. Loss berechnen
6. Backpropagation ausfuehren
7. Optimizer aktualisiert die Modellgewichte

Danach wird der naechste Batch angefordert.

### 10. Was eine komplette Epoche ist

Eine Epoche bedeutet:

- Alle im Dataset enthaltenen Bilder werden genau einmal in gemischter Reihenfolge
  durchlaufen.
- Diese Bilder werden stueckweise in Batches von 32 verarbeitet.
- Pro Batch werden die Bilder erst bei Bedarf geladen und vorverarbeitet.

Der komplette Ablauf einer Epoche ist also:

1. Indizes des Datasets mischen
2. Worker laden parallel einzelne Bilder anhand dieser Indizes
3. Worker preprocessen die Bilder einzeln
4. DataLoader baut daraus Batches zu je 32 Samples
5. Jeder Batch wird auf die GPU kopiert
6. Modell fuehrt Forward, Loss, Backward und Optimizer-Step aus
7. Wenn alle Batches verarbeitet sind, ist die Epoche beendet

### 11. Kurzfassung als Prozesskette

Die Daten durchlaufen waehrend des Trainings diese Stationen:

Festplatte  
-> Liste gueltiger Bildpfade  
-> zufaellig gemischte Sample-Indizes pro Epoche  
-> 16 Worker laden und preprocessen einzelne Bilder  
-> DataLoader stapelt 32 Bilder zu einem Batch  
-> Batch liegt im CPU-Speicher  
-> Batch wird auf die GPU kopiert  
-> Autoencoder verarbeitet den Batch  
-> Loss und Backpropagation  
-> naechster Batch

### 12. Zusammenfassung der wichtigsten Punkte

- Das Dataset laedt beim Erzeugen noch keine Bildpixel in den RAM.
- Es speichert zunaechst nur die Liste gueltiger Bilddateien.
- Die eigentlichen Bilder werden erst beim Batch-Aufbau geladen.
- Es laufen 16 Worker parallel fuer Laden und Preprocessing.
- `shuffle=True` mischt die Reihenfolge der Samples pro Epoche.
- `batch_size=32` bedeutet: 32 vorbereitete Bilder pro Trainingsschritt.
- Das Preprocessing passiert online in `__getitem__`.
- Erst der fertige Batch wird kurz vor dem Forward-Pass auf die GPU verschoben.

Um die Effizienz sauber abzuschätzen, brauche ich genau diese Informationen aus dem alten schnellen Code bzw. von der Linux-Maschine:
1. Dataset / Dateizugriff
Wie wird die Liste der Bildpfade gebaut?
Wird sie pro Run, pro Epoche oder nur einmal erstellt?
Wird glob, os.scandir, Path.glob, eine CSV/Manifest-Datei oder etwas anderes genutzt?
Werden Timestamps bei jedem Start neu geparst?
Liegen die TIFFs lokal auf SSD/NVMe oder auf Netzwerkstorage?
Sind die TIFFs komprimiert oder unkomprimiert?
2. Dataset-Klasse
Bitte zeig mir den alten Dataset/MeltDataset Code, besonders:
__init__
__len__
__getitem__
alle Hilfsfunktionen für Bildladen und Preprocessing
Das ist der wichtigste Teil, weil MLTrace aktuell pro __getitem__ noch sehr generisch durch Registry/Graph/Steps läuft.
3. Bildladen
Wurde im alten Code mit PIL, cv2.imread, tifffile.imread, imageio oder etwas anderem geladen?
Wurde direkt grayscale geladen?
Wurde uint16 -> uint8 gemacht?
Gab es Memory Mapping, Caching oder persistent geöffnete Handles?
4. Preprocessing
Für denselben Run brauche ich exakt:
Welche Schritte laufen wirklich?
Warp Perspective ja/nein?
Resize ja/nein, Zielgröße?
Crop ja/nein?
Normalize ja/nein?
CLAHE ja/nein?
Augmentation ja/nein?
Wird das Preprocessing als fertiges torchvision.transforms.Compose/Callable gebaut oder pro Bild dynamisch interpretiert?
5. DataLoader-Konfiguration
Bitte exakt aus altem Code:
batch_size
shuffle
num_workers
pin_memory
prefetch_factor
persistent_workers
drop_last
multiprocessing_context, falls gesetzt
collate_fn, falls eigene Funktion
6. Tensor-Konvertierung
Gibt __getitem__ im alten Code numpy.ndarray oder direkt torch.Tensor zurück?
Shape vor Batch: H,W, 1,H,W, oder H,W,1?
Wird float32 schon im Worker erzeugt?
Wird durch 255/65535 im Worker skaliert?
Wird .contiguous() genutzt?
7. GPU-Transfer
Wo genau passiert .to(device)?
Wird non_blocking=True genutzt?
Wird AMP genutzt?
Wird channels_last genutzt?
Wird torch.backends.cudnn.benchmark = True gesetzt?
8. Training Loop Timing
Im neuen MLTrace sollten wir Logs einbauen für:
Zeit von Worker-Start bis Dataset-Manifest fertig
Zeit bis erstes Sample geladen
Zeit bis erster Batch geladen
pro N Batches:data_wait_time
gpu_compute_time
batch_total_time
img/s
GPU device
num_workers
prefetch_factor
Batch shape/dtype

Damit kannst du im Scheduler-Log direkt sehen: wartet die GPU auf Daten oder ist das Modell selbst langsam?
9. Modellvergleich
Für den identischen Vergleich brauche ich:
alte Architektur exakt
neue MLTrace-Architektur exakt
latent_dim
Inputgröße
Batch size
Loss
Optimizer
AMP ja/nein
Validation pro Epoche ja/nein
Ein wichtiger Verdacht: MLTrace macht aktuell standardmäßig 10% Validation pro Epoche. Wenn dein alter Code keine Validation gemacht hat, kostet MLTrace pro Epoche zusätzlich ~6.000 Bilder bei 60k.
10. Linux/CUDA-Fakten
Aus der Zielmaschine:
torch.__version__
torch.version.cuda
torch.cuda.is_available()
torch.cuda.device_count()
nvidia-smi während Training
GPU utilization %
GPU memory usagelg
CPU utilization
Disk read throughput, falls möglich