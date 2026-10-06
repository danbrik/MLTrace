# STAE-3D Reconstruction 352x192

In **Methods** ist `STAE-3D Reconstruction 352x192` als gespeicherte Vorlage verfügbar und kann in **Training Pipelines** ausgewählt werden. Sie wird in neuen und beim Backendstart auch in vorhandenen Projekten ergänzt, ohne vorhandene Methoden zu ändern.

Die Architektur folgt der ausdrücklich festgelegten Rekonstruktionsvariante, inspiriert von Zhao et al. (2017), *Spatio-Temporal AutoEncoder for Video Anomaly Detection*, DOI [10.1145/3123266.3123451](https://doi.org/10.1145/3123266.3123451). Die Auflösung, BatchNorm-/Aktivierungsparameter und Trainingsvorgaben sind die projektspezifische Festlegung, keine Behauptung einer unveränderten Paper-Reproduktion.

## Auswahl und Clips

1. Normaldaten als Trainingsdatensatz und eine passende Preprocessing-Pipeline wählen. Das Ergebnis jedes Frames muss einkanalig, 352 Pixel breit und 192 Pixel hoch sein. `uint8` wird durch 255, `uint16` durch 65535 skaliert. Float-Ausgaben bleiben unverändert und sollten für Sigmoid/MSE bereits in [0,1] liegen. Training und Validierung nutzen dieselbe Verarbeitung.
2. Separate Normaldaten für die Validierung auswählen und **Beste Validierungsepoche** beibehalten. Ohne Validierung gelten weiterhin die allgemeinen Regeln: Early Stopping auf Trainingsverlust, letzte Epoche als Modell.
3. Der Datensatz-Stride bestimmt die zeitliche Ausdünnung: beispielsweise wählt Stride 5 jedes fünfte Bild der Datensatzregel. In **Methods → Sequence** verwendet diese Vorlage `ordered_index` mit `temporal_stride=1`, also jedes bereits ausgewählte Bild. Andere STAE-Methoden und explizit konfigurierte Zeitintervalle behalten ihre bisherige Auswahl.

Jeder Clip enthält 16 aufeinanderfolgende Bilder aus der bereits gefilterten Datensatzregel; benachbarte Clips überlappen um 15 Bilder. Ein fester Sekundenabstand oder eine Zeitstempel-Toleranz wird nicht erzwungen. Regeln mit weniger als 16 ausgewählten Bildern liefern keine Clips. Jede Datensatzregel wird einzeln verarbeitet; Clips überschreiten keine Regel- oder Trainings-/Validierungsgrenze. Dieselbe Auswahl wird im Dry Run, Training und in der Inferenz verwendet. Beim Backendstart werden nur unveränderte gespeicherte Exemplare dieser Vorlage auf die neue Auswahl umgestellt; individuell angepasste Methoden bleiben erhalten.

## Architektur

Tensorformat `(N,C,T,H,W)`, Input `(N,1,16,192,352)`. Encoderkanäle 32,48,64,64 mit 3×3×3-Conv, Stride 1, Padding 1; nach den ersten drei Blöcken MaxPool 2. Bottleneck `(N,64,2,24,44)`. Decoder: drei ConvTranspose3d mit 48,32,32 Kanälen, Kernel 3, Stride 2, Padding 1, Output Padding 1; abschließend Conv3d nach 1 Kanal und Sigmoid. Alle sieben versteckten Blöcke verwenden BatchNorm3d (eps 0,001; momentum 0,01) und LeakyReLU (negative_slope 0,3). Conv-Bias ist aktiv, Initialisierung durch PyTorch. Keine Dense-Layer und kein Vorhersagezweig.

`app.modeling.stae_reconstruction.build_reconstruction_model(torch)` liefert ein PyTorch-Modul mit `forward(x)` → Rekonstruktion und `encode(x)` → Bottleneck. Derselbe gespeicherte Graph wird vom bestehenden Trainings-/Inferenzadapter verwendet; dessen internes Rückgabeformat bleibt kompatibel.

## Training

Adam mit Lernrate 0,0001 und ansonsten PyTorch-Defaults (Weight Decay 0), MSE, maximal 1000 Epochen, Batchgröße 4, Trainings-Shuffle aktiv. Bei unzureichendem Speicher kann die Batchgröße in Training Pipelines auf 2 reduziert werden; keine stille Änderung laufender Trainings. Tatsächliche Batchgröße, Gerät und Seed werden protokolliert. Seed 0 initialisiert `torch`, `numpy` und `random`. CUDA wenn verfügbar, sonst CPU; AMP standardmäßig aus.

Early Stopping: Patience 5, `best_stop_loss - current_loss > 1e-8`. Die minimale Verbesserung steuert den Geduldszähler; die Modellwahl verwendet unabhängig davon den tatsächlich niedrigsten Validierungsverlust. Bei gleichen Verlusten bleibt die frühere Epoche gewählt. Checkpoints sichern letzten Trainings-/Optimizer-/Zufallszustand und beste Modellgewichte einschließlich BatchNorm-Puffern. Nach erfolgreichem regulärem Ende oder Early Stopping werden die besten Gewichte exportiert. Ein manueller Abbruch erzeugt kein fertiges Modell und bleibt fortsetzbar.

Der Anomalie-Score ist standardmäßig der mittlere quadratische Rekonstruktionsfehler; es wird kein Prediction-Loss berechnet. Das Modell sollte ausschließlich auf Normaldaten trainiert werden.
