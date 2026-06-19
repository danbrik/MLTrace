# Legacy Algorithm

## Zweck

Diese Doku beschreibt den alten Trainingspfad so konkret wie moeglich.
Ziel ist ein fairer Vergleich gegen MLTrace, vor allem fuer:

- Dataset-Aufbau
- Dateizugriff
- Preprocessing
- DataLoader-Verhalten
- GPU-Transfer
- Training-Loop-Overhead

Wichtig:
Diese Antworten basieren auf dem vorhandenen Legacy-Code und der aktuellen Konfiguration.
Wo Informationen nicht sicher aus dem Code ableitbar sind, sind sie als offen markiert.

## Kurzfazit

Der alte Code ist relativ direkt und leichtgewichtig aufgebaut:

- Bildpfade werden einmal beim Dataset-Aufbau gesammelt.
- Pro Epoche wird nur die Reihenfolge der bereits bekannten Samples gemischt.
- Pro __getitem__ wird genau ein TIFF geladen und inline vorverarbeitet.
- Es gibt keine Registry, keinen generischen Step-Graph und kein sichtbares Dataset-Manifest-Caching.
- Das Modell trainiert ohne Validation-Loop, sofern kein X_val uebergeben wird.
- Die GPU bekommt jeweils den fertigen Batch, nicht einzelne Bilder fruehzeitig.

## 1. Dataset / Dateizugriff

### Frage
Wie wird die Liste der Bildpfade gebaut?

### Antwort
Die Bildpfadliste wird direkt aus dem Dateisystem aufgebaut.

Im aktuellen Legacy-Code gilt:

- Eine YAML-Konfiguration liefert den Basisordner fuer den gewaehlten Tank.
- Bei folder_structure='flat' werden alle TIFF-Dateien per Path.glob("*.tiff") gesucht.
- Bei folder_structure='hierarchical' werden Tagesordner iteriert und darin jeweils TIFFs gesucht.
- Danach werden die gefundenen Pfade anhand ihres Zeitstempels gefiltert.
- Anschliessend werden optional ausgeschlossene Zeitraeume entfernt.
- Danach wird sampling_rate ueber List-Slicing angewendet.

Es gibt keine CSV, kein Manifest und keine vorgelagerte Index-Datei.

### Frage
Wird die Liste pro Run, pro Epoche oder nur einmal erstellt?

### Antwort
Die Liste wird einmal pro Dataset-Instanz erstellt.

Das bedeutet:

- pro Training-Run: ja
- pro Dataset-Erzeugung: ja
- pro Epoche: nein

Sobald das Dataset gebaut ist, lebt die Liste in self.image_paths.
Die Epoche arbeitet danach nur noch mit Indizes auf diese Liste.

### Frage
Wird glob, os.scandir, Path.glob, CSV/Manifest oder etwas anderes genutzt?

### Antwort
Es wird pathlib.Path.glob genutzt.

Konkret:

- Path.glob("*.tiff") fuer die flache Struktur
- date_dir.glob("*.tiff") fuer die hierarchische Struktur
- fuer die Tagesordner selbst wird iterdir() verwendet

### Frage
Werden Timestamps bei jedem Start neu geparst?

### Antwort
Ja, beim Dataset-Aufbau werden die Timestamps aus den Dateinamen neu geparst.

Das passiert, weil fuer das Filtern nach Zeitfenster und Ausschlussintervallen jede Datei geprueft wird.

Zusaetzlich gilt:

- wenn apply_camera_calibration=False, dann endet das Timestamp-Parsing praktisch nach dem Dataset-Aufbau
- wenn apply_camera_calibration=True, dann wird im __getitem__ pro Sample der Timestamp erneut aus dem Dateinamen gelesen, um die passende Homographie-Matrix auszuwaehlen

### Frage
Liegen die TIFFs lokal auf SSD/NVMe oder auf Netzwerkstorage?

### Antwort
Nach der aktuellen Pfadkonfiguration liegen die Daten sehr wahrscheinlich auf Netzwerkstorage.

Die Tank-Pfade zeigen auf Mountpoints unter:

- /net/fileserver1/...

Das spricht klar gegen lokale NVMe-Daten und fuer ein gemountetes Netzlaufwerk oder Fileserver-Storage.

### Frage
Sind die TIFFs komprimiert oder unkomprimiert?

### Antwort
Das ist aus dem Python-Code allein nicht sicher ableitbar.

Der Code oeffnet TIFFs generisch, aber liest nirgends TIFF-Metadaten wie Compression.
Dafuer braucht man eine Dateipruefung auf der Zielmaschine, zum Beispiel mit:

- tiffinfo
- identify -verbose
- python plus PIL oder tifffile

Aktueller Status:
offen.

## 2. Dataset-Klasse

### Frage
Wie sieht die alte MeltDataset-Klasse aus?

### Antwort
Die Klasse ist klassisch und direkt aufgebaut: Pfadliste in __init__, Anzahl ueber __len__, Laden plus Preprocessing in __getitem__.

### Wichtige Eigenschaften

- kein Manifest-Cache
- kein globales Preprocessing-Graph-Objekt
- keine persistent offenen TIFF-Handles
- keine Memory-Mapped Bilddaten
- keine Offline-Preprocessing-Artefakte fuer das Training
- alles laeuft inline im Dataset

### __init__

```python
class MeltDataset(Dataset):
    def __init__(
        self,
        config_path,
        tank="hf_w14",
        folder_structure="hierarchical",
        mode="train",
        start_date=None,
        end_date=None,
        sampling_rate=1,
        augment_brightness=False,
        augment_rotation=False,
        clahe=False,
        transformation=[],
        apply_camera_calibration=False,
        center_crop=True,
        convert_to_8bit=True,
        minmax=True,
        exclude_timeranges=None,
        resize_to=512,
        verbose=True,
        train_mode="timerange",
        timepoints=None,
        include_range_minutes=None
    ):
        self.tank = tank
        self.sampling_rate = sampling_rate
        self.augment_brightness = augment_brightness
        self.augment_rotation = augment_rotation
        self.clahe = clahe
        self.center_crop = center_crop
        self.convert_to_8bit = convert_to_8bit
        self.transformation = transformation
        self.apply_camera_calibration = apply_camera_calibration
        self.minmax = minmax
        self.exclude_timeranges = exclude_timeranges or []
        self.resize_to = resize_to
        self.verbose = verbose
        self.train_mode = str(train_mode or "timerange").lower().strip()
        self.timepoints = self._normalize_timepoints(timepoints)
        self.include_range_minutes = include_range_minutes

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.time_dependent_transformations = []
        if self.apply_camera_calibration:
            self.time_dependent_transformations = config.get('time_dependent_transformations', [])
            for t in self.time_dependent_transformations:
                t['start_time'] = datetime.datetime.strptime(t['start_time'], '%Y-%m-%d %H:%M')
                t['end_time'] = datetime.datetime.strptime(t['end_time'], '%Y-%m-%d %H:%M')
                t['matrix'] = np.array(t['matrix'], dtype=np.float32)

        base_dir = Path(config[tank])

        if folder_structure == "flat":
            self.image_paths = sorted(base_dir.glob("*.tiff"))

            if start_date or end_date:
                filtered_paths = []
                for img_path in self.image_paths:
                    timestamp = self._extract_timestamp_from_filename(img_path)
                    if timestamp is not None:
                        if self._is_in_selected_range(timestamp, start_date, end_date):
                            if not self._is_in_excluded_range(timestamp):
                                filtered_paths.append(img_path)
                self.image_paths = filtered_paths
        else:
            self.root_dir = base_dir / mode
            self.image_paths = []
            for date_dir in sorted(self.root_dir.iterdir()):
                try:
                    date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                    if self._date_dir_is_relevant(date, start_date, end_date):
                        day_images = sorted(date_dir.glob("*.tiff"))
                        if self.exclude_timeranges or self.train_mode == "timepoints" or start_date or end_date:
                            filtered_day_images = []
                            for img_path in day_images:
                                timestamp = self._extract_timestamp_from_filename(img_path)
                                if timestamp is not None and self._is_in_selected_range(timestamp, start_date, end_date) and not self._is_in_excluded_range(timestamp):
                                    filtered_day_images.append(img_path)
                            self.image_paths.extend(filtered_day_images)
                        else:
                            self.image_paths.extend(day_images)
                except ValueError:
                    continue

        self.image_paths = self.image_paths[::sampling_rate]
```

### __len__

```python
def __len__(self):
    return len(self.image_paths)
```

### __getitem__

```python
def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    img_path = self.image_paths[idx]

    try:
        im = Image.open(img_path)

        if self.convert_to_8bit:
            im = tiff_force_8bit(im)

        if self.apply_camera_calibration and self.time_dependent_transformations:
            timestamp = self._extract_timestamp_from_filename(img_path)
            matrix = self._get_time_dependent_matrix(timestamp)
            if matrix is not None:
                im_np = np.array(im)
                im_np = cv2.warpPerspective(im_np, matrix, (1920, 1200), borderMode=cv2.BORDER_CONSTANT)
                im = Image.fromarray(im_np)

        augmentation = self.augment_brightness | self.augment_rotation
        im_tensor = torch_preprocessing_pipeline(
            im,
            augmentation,
            self.augment_rotation,
            center_crop=self.center_crop,
            transformation=self.transformation,
            resize_to=self.resize_to,
        )

        if self.transformation != []:
            im_tensor = perspective_transform(im_tensor, self.transformation, resize_to=self.resize_to)

        if im_tensor.dtype == torch.uint8:
            im_tensor = im_tensor.to(torch.float32) / 255.0
        elif im_tensor.dtype == torch.uint16:
            im_tensor = im_tensor.to(torch.float32) / 65535.0
        else:
            im_tensor = im_tensor.to(torch.float32)

        if self.clahe:
            im_tensor = apply_clahe(im_tensor)

        if self.minmax:
            tensor_min = im_tensor.min()
            tensor_max = im_tensor.max()
            if tensor_max > tensor_min:
                im_tensor = (im_tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                im_tensor = torch.zeros_like(im_tensor)

        return im_tensor

    except Exception:
        return torch.full((1, self.resize_to, self.resize_to), float('nan'))
```

### Hilfsfunktionen fuer Bildladen und Preprocessing

Fuer die Bildauswahl und Vorverarbeitung sind im Legacy-Code besonders diese Hilfsfunktionen relevant:

- _normalize_timepoints
- _is_in_selected_range
- _date_dir_is_relevant
- _extract_timestamp_from_filename
- _extract_timestamp_hf_w14
- _extract_timestamp_planet
- _is_in_excluded_range
- _get_time_dependent_matrix
- tiff_force_8bit
- torch_preprocessing_pipeline
- perspective_transform
- apply_clahe

### tiff_force_8bit

```python
def tiff_force_8bit(image):
    if image.format == 'TIFF' and image.mode == 'I;16':
        array = np.array(image)
        normalized = (array.astype(np.uint16) - array.min()) * (255.0) / (array.max() - array.min())
        image = Image.fromarray(normalized.astype(np.uint8))
    return image
```

### torch_preprocessing_pipeline

```python
def torch_preprocessing_pipeline(
    image,
    augmentation=True,
    rotation=True,
    center_crop=True,
    transformation=[],
    resize_to=512,
):
    image_tensor = v2.PILToTensor()(image)
    pipe_list = []

    if augmentation:
        if rotation:
            pipe_list += [v2.RandomRotation(degrees=5)]
        if image_tensor.dtype == torch.uint16:
            image_tensor = image_tensor.to(torch.float32) / 65535.0
            pipe_list += [v2.ColorJitter(brightness=(0.5, 1.3))]
            if pipe_list:
                preproc_pipe = v2.Compose(pipe_list)
                image_tensor = preproc_pipe(image_tensor)
                pipe_list = []
            image_tensor = (image_tensor * 65535.0).to(torch.uint16)
        else:
            pipe_list += [v2.ColorJitter(brightness=(0.5, 1.3))]

    if center_crop:
        pipe_list += [v2.CenterCrop(990)]

    if transformation == []:
        pipe_list += [v2.Resize([resize_to, resize_to])]

    if pipe_list:
        preproc_pipe = v2.Compose(pipe_list)
        image_tensor = preproc_pipe(image_tensor)

    return image_tensor
```

### perspective_transform

```python
def perspective_transform(image_tensor, transformation, resize_to=512):
    if not transformation or len(transformation) != 4:
        return image_tensor

    original_dtype = image_tensor.dtype

    if image_tensor.ndim == 3:
        if image_tensor.shape[0] == 1:
            image_np = image_tensor.squeeze(0).numpy()
        else:
            image_np = image_tensor.permute(1, 2, 0).numpy()
    elif image_tensor.ndim == 2:
        image_np = image_tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    image_np = image_np.astype(np.uint16 if original_dtype == torch.uint16 else np.uint8)

    src_pts = order_points(transformation)
    (tl, tr, br, bl) = src_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    square_size = max(max_width, max_height, 1)

    dst_pts = np.array([
        [0, 0],
        [square_size - 1, 0],
        [square_size - 1, square_size - 1],
        [0, square_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        image_np,
        M,
        (square_size, square_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if warped.ndim == 2:
        warped_tensor = torch.from_numpy(warped).unsqueeze(0)
    else:
        warped_tensor = torch.from_numpy(warped).permute(2, 0, 1)

    warped_tensor = v2.Resize([resize_to, resize_to])(warped_tensor)
    return warped_tensor
```

### apply_clahe

```python
def apply_clahe(image_tensor):
    if image_tensor.ndim == 2:
        image_np = image_tensor.numpy()
    elif image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
        image_np = image_tensor.squeeze(0).numpy()
    elif image_tensor.ndim == 3:
        image_np = image_tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    image_np = (image_np * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if image_np.ndim == 2:
        clahe_image = clahe.apply(image_np)
        clahe_image_tensor = torch.from_numpy(clahe_image).unsqueeze(0).to(torch.float32) / 255.0
    else:
        channels = cv2.split(image_np)
        clahe_channels = [clahe.apply(channel) for channel in channels]
        clahe_image = cv2.merge(clahe_channels)
        clahe_image_tensor = torch.from_numpy(clahe_image).permute(2, 0, 1).to(torch.float32) / 255.0

    return clahe_image_tensor
```

## 3. Bildladen

### Frage
Wurde im alten Code mit PIL, cv2.imread, tifffile.imread, imageio oder etwas anderem geladen?

### Antwort
Im Trainings-Dataset wird mit PIL geladen.

Genauer:

- Image.open(img_path) fuer das eigentliche Training
- cv2.imread(...) existiert nur in einer separaten Hilfsfunktion fuer Graustufenladen, ist aber nicht der Standardpfad des Trainings-Datasets
- tifffile.imread wird im Trainingspfad nicht verwendet
- imageio wird im Trainingspfad nicht verwendet

### Frage
Wurde direkt grayscale geladen?

### Antwort
Nicht explizit per Loader-Flag.

Der Trainingspfad nimmt das TIFF so, wie PIL es oeffnet.
Anschliessend wird es per PILToTensor() in einen Tensor ueberfuehrt.

Praktisch ist das Training auf 1 Kanal ausgelegt, aber der Loader erzwingt beim Oeffnen nicht per convert("L") eine Graustufe.

### Frage
Wurde uint16 -> uint8 gemacht?

### Antwort
Optional, ja.

Wenn convert_to_8bit=True, wird ueber tiff_force_8bit(...) konvertiert.
Wenn convert_to_8bit=False, bleibt das Sample zunaechst in 16-bit und wird spaeter direkt nach float32 in [0,1] skaliert.

Fuer den aktuellen Trainingslauf gilt:

- convert_to_8bit=False

### Frage
Gab es Memory Mapping, Caching oder persistent geoeffnete Handles?

### Antwort
Nein, nichts davon ist im Legacy-Trainingspfad sichtbar.

Es gibt:

- kein Memory Mapping
- kein File-Handle-Reuse
- kein Bildcache im Dataset
- kein Manifest-Cache fuer Sample-Metadaten
- keine persistent geoeffneten TIFF-Dateien

Pro Sample gilt schlicht:

- Pfad holen
- Datei oeffnen
- preprocessen
- Tensor zurueckgeben

## 4. Preprocessing

### Frage
Welche Schritte laufen fuer denselben Run wirklich?

### Antwort
Fuer den aktuellen Trainingslauf mit den gezeigten Parametern gilt:

- Warp Perspective: ja
- Resize: ja
- Crop: nein
- Skalierung auf [0,1]: ja
- Min-Max-Normalisierung: nein
- CLAHE: nein
- Augmentation: nein
- Kamera-Kalibrierung: nein
- 16-bit zu 8-bit: nein

### Aktuelle Konfiguration des gezeigten Runs

- augment_brightness=False
- augment_rotation=False
- clahe=False
- apply_camera_calibration=False
- convert_to_8bit=False
- center_crop=False
- minmax=False
- resize_to=840
- transformation=[[42, 44], [885, 44], [895, 630], [53, 630]]

### Frage
Warp Perspective ja oder nein?

### Antwort
Ja.

Da transformation nicht leer ist, wird perspective_transform(...) ausgefuehrt.

### Frage
Resize ja oder nein, Zielgroesse?

### Antwort
Ja, Zielgroesse ist 840 x 840.

Wichtig:
Bei aktivierter Perspektivtransformation passiert das Resize nicht im ersten Compose-Pfad, sondern innerhalb von perspective_transform(...).

### Frage
Crop ja oder nein?

### Antwort
Nein.

center_crop=False, daher kein CenterCrop.

### Frage
Normalize ja oder nein?

### Antwort
Es gibt zwei verschiedene Normalisierungsarten, die man auseinanderhalten muss:

1. Immer vorhandene Skalierung auf [0,1]
   - ja
2. Zusaetzliche per-Bild Min-Max-Normalisierung
   - nein, weil minmax=False

### Frage
CLAHE ja oder nein?

### Antwort
Nein.

### Frage
Augmentation ja oder nein?

### Antwort
Nein.

Weder Helligkeitsaugmentation noch Rotationsaugmentation laufen im aktuellen Run.

### Frage
Wird das Preprocessing als fertiges torchvision.transforms.Compose oder Callable gebaut oder pro Bild dynamisch interpretiert?

### Antwort
Es wird pro Bild dynamisch aufgebaut.

Das ist wichtig fuer den Vergleich gegen MLTrace:

- Es gibt kein einmalig in __init__ gebautes, festes Preprocessing-Objekt fuer den gesamten Run.
- Stattdessen wird in torch_preprocessing_pipeline(...) bei jedem __getitem__ eine lokale pipe_list aufgebaut.
- Daraus wird bei Bedarf ein v2.Compose(...) erzeugt und sofort auf dieses eine Sample angewendet.

## 5. DataLoader-Konfiguration

### Frage
Bitte exakt aus altem Code: batch_size, shuffle, num_workers, pin_memory, prefetch_factor, persistent_workers, drop_last, multiprocessing_context, collate_fn

### Antwort
Fuer das Training im Legacy-Code:

- batch_size = 32
- shuffle = True
- num_workers = 16
- pin_memory = True
- prefetch_factor = nicht explizit gesetzt, also PyTorch-Default
- persistent_workers = nicht explizit gesetzt, also PyTorch-Default
- drop_last = nicht explizit gesetzt, also False per Default
- multiprocessing_context = nicht gesetzt
- collate_fn = nicht gesetzt, Standard-Collate von PyTorch

### Frage
Wie viele Worker laufen also wirklich?

### Antwort
Im Training laufen 16 Worker-Prozesse parallel.

Diese Worker:

- laden die Bilddateien
- fuehren __getitem__ aus
- geben fertige Sample-Tensoren an den DataLoader zurueck

Sie trainieren nicht das Modell selbst.
Das Training laeuft im Hauptprozess.

## 6. Tensor-Konvertierung

### Frage
Gibt __getitem__ im alten Code numpy.ndarray oder direkt torch.Tensor zurueck?

### Antwort
Direkt torch.Tensor.

### Frage
Shape vor Batch: H,W, 1,H,W, oder H,W,1?

### Antwort
Im normalen Trainingspfad:

- 1 x H x W

Fuer den aktuellen Run nach Resize:

- 1 x 840 x 840

### Frage
Wird float32 schon im Worker erzeugt?

### Antwort
Ja.

Die Konvertierung zu float32 passiert im __getitem__, also im DataLoader-Worker.

### Frage
Wird durch 255 oder 65535 im Worker skaliert?

### Antwort
Ja.

Die Skalierung passiert ebenfalls im __getitem__.

Logik:

- wenn Tensor uint8, dann /255.0
- wenn Tensor uint16, dann /65535.0

Fuer den aktuellen Run ist wegen convert_to_8bit=False der wahrscheinliche Pfad:

- TIFF in 16-bit laden
- nach allen geometrischen Schritten in float32
- Division durch 65535.0

### Frage
Wird .contiguous() genutzt?

### Antwort
Im Legacy-Trainingspfad nicht sichtbar.

Es gibt keine explizite Verwendung von .contiguous() im Datenpfad.

## 7. GPU-Transfer

### Frage
Wo genau passiert .to(device)?

### Antwort
An zwei Stellen:

1. Das Modell selbst wird direkt nach der Initialisierung auf das Device gelegt.
2. Jeder Trainingsbatch wird im Trainingsloop auf das Device verschoben.

### Modell-Transfer

Das Modell wird im Konstruktor des Autoencoders direkt auf das Ziel-Device geschoben.

### Batch-Transfer

Im Trainingsloop passiert fuer jeden Batch:

```python
data = data.to(self.device, non_blocking=True)
```

Das ist der zentrale GPU-Transfer fuer Trainingsdaten.

### Frage
Wird non_blocking=True genutzt?

### Antwort
Ja.

### Frage
Wird AMP genutzt?

### Antwort
Nein.

Es gibt im Legacy-Trainingspfad keinen sichtbaren Einsatz von:

- torch.autocast
- GradScaler
- mixed precision utilities

### Frage
Wird channels_last genutzt?

### Antwort
Nein, im sichtbaren Legacy-Code nicht.

### Frage
Wird torch.backends.cudnn.benchmark = True gesetzt?

### Antwort
Im sichtbaren Legacy-Code nicht.

## 8. Training Loop Timing

### Frage
Welche Logs sollten im neuen MLTrace eingebaut werden?

### Antwort
Fuer einen fairen Effizienzvergleich sollten mindestens diese Metriken geloggt werden.

### Empfohlene Start-Metriken

- Zeit vom Prozessstart bis Dataset-Manifest fertig
- Zeit vom Dataset-Bau bis erstes Sample geladen
- Zeit bis erster Batch fertig
- Zeit bis erster GPU-Forward startet
- Zeit bis erster Optimizer-Step fertig

### Empfohlene Laufzeit-Metriken pro N Batches

- data_wait_time
- gpu_compute_time
- batch_total_time
- images_per_second
- batch_shape
- batch_dtype
- device_name
- num_workers
- prefetch_factor
- pin_memory
- persistent_workers
- cpu_utilization
- gpu_utilization
- gpu_memory_used

### Interpretationsregel

Wenn data_wait_time hoch ist und gpu_compute_time niedrig, dann wartet die GPU auf Daten.

Wenn gpu_compute_time dominiert und data_wait_time klein ist, dann ist eher das Modell oder der GPU-Teil selbst der Engpass.

## 9. Modellvergleich

### Frage
Wie sieht die alte Architektur exakt aus?

### Antwort
Die alte Architektur ist ein CNN-Autoencoder mit:

- Input-Kanaele: 1
- Input-Groesse: 840 x 840
- Latent-Dimension: 200
- Encoder-Kanaele: [32, 64, 128]
- batchnorm=False
- maxpooling=False

### Exakter Encoder

Der Encoder baut drei Convolution-Bloecke mit stride=2, kernel_size=3, padding=1.

Mit aktueller Konfiguration ergibt sich:

- Conv2d 1 -> 32
- Conv2d 32 -> 64
- Conv2d 64 -> 128
- danach Flatten
- danach Linear auf 200

Da image_size=840 und drei Downsampling-Schritte mit Faktor 2 vorliegen:

- 840 -> 420 -> 210 -> 105

Der Flatten-Vektor hat also:

- 128 * 105 * 105 = 1411200

Darauf folgt:

- Linear(1411200, 200)

### Exakter Decoder

Der Decoder macht:

- Linear(200, 128 * 105 * 105)
- Reshape auf 128 x 105 x 105
- dann drei ConvTranspose2d-Bloecke
- am Ende Conv2d auf 1 Kanal mit Sigmoid

Mit aktueller Logik entsteht:

- ConvTranspose2d 128 -> 128
- ConvTranspose2d 128 -> 64
- ConvTranspose2d 64 -> 32
- Final Conv2d 32 -> 1
- Sigmoid

### Frage
Wie sieht die neue MLTrace-Architektur exakt aus?

### Antwort
Dazu gibt es in diesem Workspace keine belastbare Quelle.

Aktueller Status:
offen.

Fuer einen fairen Vergleich muss fuer MLTrace separat dokumentiert werden:

- exakte Layer-Struktur
- Input-Shape
- Latent-Dimension
- Loss
- Optimizer
- AMP ja oder nein
- Validation-Strategie
- Preprocessing-Pfad
- Loader-Konfiguration

### Frage
Wie sind die alten Trainingsparameter fuer den gezeigten Run?

### Antwort
Fuer den gezeigten Legacy-Run:

- latent_dim = 200
- input_size = 840
- batch_size = 32
- loss = MSELoss
- optimizer = Adam
- learning_rate = 0.0008024259379673648
- epochs = 100
- AMP = nein
- Validation pro Epoche = nein

### Frage
Gab es Validation pro Epoche?

### Antwort
Im normalen Legacy-Trainingspfad nein.

Der fit(...)-Code kann zwar validieren, aber nur wenn X_val uebergeben wird.
Im gezeigten Trainingsaufruf wird nur train_loader uebergeben und kein Validation-Dataset.

Das ist fuer den Vergleich sehr wichtig:

- alter Code: keine Validation pro Epoche
- wenn MLTrace standardmaessig 10 Prozent Validation pro Epoche macht, ist der Vergleich sonst nicht fair

## 10. Linux / CUDA Fakten

### Frage
Welche Linux- und CUDA-Fakten sind aus dem Code sicher bekannt?

### Antwort
Aus dem Code sicher bekannt:

- es wird PyTorch genutzt
- CUDA wird nur verwendet, wenn torch.cuda.is_available() wahr ist
- bei mehreren GPUs wird die GPU mit dem meisten freien Speicher per NVML ausgewaehlt

### Frage
Welche Laufzeitfakten sind aus dem Workspace nicht sicher beantwortbar?

### Antwort
Nicht sicher aus dem vorhandenen Code ableitbar sind:

- torch.__version__
- torch.version.cuda
- torch.cuda.device_count()
- reale GPU-Auslastung waehrend Training
- reale GPU-Memory-Nutzung
- CPU-Auslastung
- Disk-Read-Throughput
- nvidia-smi-Momentaufnahme
- TIFF-Kompression auf Datei-Ebene

Diese Punkte muessen direkt auf der Zielmaschine erhoben werden.

### Empfohlene Commands auf der Zielmaschine

```bash
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

```bash
nvidia-smi
```

```bash
nvidia-smi dmon
```

```bash
top
```

```bash
iostat -xz 1
```

```bash
pidstat -dru 1
```

### Empfohlene Zusatzpruefung fuer TIFF-Kompression

```bash
python - <<'PY'
from PIL import Image
img = Image.open("beispiel.tiff")
print(img.format, img.mode, img.size)
print(img.tag_v2)
PY
```

Oder mit tifffile:

```bash
python - <<'PY'
import tifffile
with tifffile.TiffFile("beispiel.tiff") as tif:
    page = tif.pages[0]
    print("compression =", page.compression)
    print("dtype =", page.dtype)
    print("shape =", page.shape)
PY
```

## Vergleichsrelevante Kernpunkte gegen MLTrace

Die wichtigsten Legacy-Eigenschaften fuer den Performancevergleich sind:

- Pfadliste wird einmal pro Dataset-Instanz erstellt
- kein Manifest-Cache
- Timestamps werden beim Dataset-Aufbau aus Dateinamen geparst
- Dateisuche ueber Path.glob
- Training nutzt PIL-basiertes Laden
- Preprocessing laeuft pro Sample inline im Worker
- keine AMP
- keine Validation im Standard-Training
- num_workers=16
- batch_size=32
- pin_memory=True
- GPU-Transfer mit non_blocking=True
- Netzwerkstorage unter /net/fileserver1/... ist sehr wahrscheinlich beteiligt

## Offene Punkte, die fuer den fairen Vergleich noch aufgeloest werden muessen

- reale TIFF-Kompression
- reale Storage-Charakteristik der Zielmaschine
- MLTrace-Architektur exakt
- MLTrace-Validation-Policy exakt
- MLTrace-Loader-Konfiguration exakt
- MLTrace-AMP-Verhalten
- reale GPU- und CPU-Auslastung unter Last
- First-batch-Latency alt vs. neu
- Throughput alt vs. neu in img/s

## Minimaler Vergleichscheck alt vs. MLTrace

Fuer einen fairen A/B-Vergleich sollten beide Seiten identisch haben:

- gleiche Bildmenge
- gleiche Inputgroesse
- gleiches Resize-Verhalten
- gleiche Perspective-Warp-Logik
- gleiche Skalierung auf [0,1]
- keine zusaetzliche Validation, falls Legacy keine hatte
- gleiche Batchgroesse
- gleiche Anzahl Worker
- gleiche GPU
- AMP auf beiden Seiten entweder an oder aus
- identische Architektur oder klar dokumentierte Abweichung

## Endfazit

Der Legacy-Code ist vor allem deshalb schnell vergleichbar, weil er relativ direkt arbeitet:

- Dateiliste aufbauen
- pro Sample TIFF laden
- wenige feste Preprocessing-Schritte
- Tensor im Worker erzeugen
- Batch bilden
- Batch auf GPU
- trainieren

Der groesste Unterschied zu einem generischen MLTrace-Pfad wird sehr wahrscheinlich nicht im eigentlichen CNN liegen, sondern in:

- Dataset- oder Manifest-Aufbau
- Abstraktionskosten pro __getitem__
- First-batch-Latency
- Validation-Overhead
- allgemeinem Scheduling- und Pipeline-Overhead