# Zeitreihenmodelle, Training und Ergebnisse

Im Arbeitsbereich **Zeitreihen** folgen auf Datenbasis und Splits die Seiten **Models**, **Training Pipelines** und **Results**. Die Erweiterung verwendet PyTorch aus `pip install -e '.[ml]'`. Migration `0059_time_series_training` ergänzt ausschließlich neue Tabellen; die vorhandenen Bilddatenfunktionen bleiben erhalten. Projektmigrationen laufen beim Öffnen über den bestehenden Projektmechanismus.

## Bedienung

1. CSV unter Datenbasis einlesen und Sensorspalten wählen. Einen zeitlichen Split mit Train, optional Validation und Test speichern. Tags annotieren diese Zeiträume; sie filtern keine Trainingsdaten.
2. Unter Models die Architekturvorlage öffnen oder duplizieren. Jeder editierbare Parameter zeigt Default, Herkunft, Quelle und Überschreibung. Gelöschte vorbelegte Modelle werden nicht automatisch wieder sichtbar; neue Varianten lassen sich weiter aus der Architekturdefinition anlegen.
3. Unter Training Pipelines Datenbasis, Split und Modell wählen. Die Preview nennt tatsächlich verwendete Zeilen, vollständige Fenster, Anlaufpunkte, Sensoranzahl, Samplingabstand, Zeitlücken und Encoderform. Eine neue Pipeline schlägt die nächste ganze Anzahl Samples für 180 Minuten nominale Historie vor. Manuell eingegebene und gespeicherte L bleiben erhalten, auch beim Modellwechsel.
4. Speichern übernimmt den aktuellen Split und die aktuelle Architektur. **Gespeicherten Stand starten** verwendet exakt den eingefrorenen Pipelinestand. **Speichern und Training starten** übernimmt vorher Änderungen. Bereits gestartete Läufe bleiben unverändert.
5. Trainingsstatus, Epochen, Loss, Laufzeit, Logs und Abbrechen stehen in der Pipeline und im gemeinsamen Scheduler bereit. Results zeigt abgeschlossene Läufe, Sensorrekonstruktionen, beide Scores, Tags, Anlaufpunkte, Trainingskurven, Skalierung und den vollständigen Snapshot. Große Reihen sind mit 5.000 Ergebnisendpunkten pro Seite zugänglich; CSV und Repräsentationsexport enthalten alle Endpunkte.

Bei fünf Minuten Sampling bedeutet **L=36**: 180 Minuten nominale Historie, aber 175 Minuten zwischen erstem und letztem Zeitstempel. Schrittweite ist ein Sample. Ein Fenster liefert sein Ergebnis am letzten Zeitstempel. Zeitstempel bleiben in den Artefakten und API-Antworten nanosekundengenau in UTC; die Diagrammdarstellung verwendet die Zeitauflösung des Browsers.

## Datenvertrag

Die gemeinsame Aufbereitung liegt in `backend/app/time_series/data.py`. Sie sortiert nach Zeit, erhält die ausgewählte Sensorreihenfolge und bestimmt Min/Max ausschließlich auf zugeordneten Train-Zeilen **vor** der Fensterbildung. Validation und Test nutzen dieselben Parameter. Kein Clipping: Werte kleiner 0 oder größer 1 bleiben erhalten. Konstante Train-Spalten erhalten Nenner 1; Namen und Parameter stehen im Snapshot und in der Preview.

Sampling ist der Median positiver Abstände benachbarter Train-Zeilen. Abstände größer als 1,5 × Median, Wechsel der Gruppe und ausgeschlossene Zeilen beginnen eine neue Sequenz. Tagwechsel innerhalb derselben Gruppe unterbrechen keine Fenster. Es wird weder resampelt noch interpoliert. Die ersten L−1 Zeilen jeder Sequenz besitzen weder Score noch Ersatzlatent. Train und Test brauchen mindestens ein vollständiges Fenster; konfigurierte Validation ebenfalls.

## Modelle und Quellen

Die versionierte Registry `definitions.py` trennt **Paper**, **Autorenimplementierung** und **MLTrace-Standard** auf Feldebene. Die Eingabedefinition, Architektur, Trainingsparameter und Benutzerüberschreibungen werden mit dem Run gespeichert.

- **USAD**: gemeinsamer MLP-Encoder, zwei Sigmoid-Decoder; zwei Adam-Optimierer mit neu berechnetem Graph pro Schritt und epochenabhängigen adversarialen Gleichungen. Referenz: Audibert et al., DOI 10.1145/3394486.3403392, Gleichungen 7–9 und Architekturanhang, SWaT-Profil. Latent 40 und 70 Epochen stammen aus diesem Profil; Batch 128 und Scoregewichte 0,5/0,5 sind MLTrace-Defaults.
- **TCN-AE**: Residual-TCN mit Skip-Summen, 1×1-Projektion, Average Pooling, wiederholendes Upsampling und linearer Rekonstruktion. Referenz: `MarkusThill/bioma-tcn-ae/src/tcnae.py`, BIOMA-Baseline. **MLTrace-Architekturanpassung**: Pooling 6 statt 42. L=36 ergibt 6×8=48 Encoderwerte. Partielle letzte Poolinggruppen werden mit ihrem tatsächlichen Umfang gemittelt; die hochgesampelte Ausgabe wird auf L zugeschnitten. Pooling größer als L führt zu einem Fehler, nicht zur Änderung von L. Log-Cosh/AMSGrad dienen dem Training; der AD-Score ist direktes MSE ohne die nachträgliche Test-Anpassung und Glättung der Autorenimplementierung.
- **LSTM-VAE**: ausdrücklich **adaptierte Fenster-Variante**, keine exakte Reproduktion von Park et al., arXiv:1711.00614v1. Posteriorparameter entstehen pro LSTM-Schritt, der Decoder rekonstruiert Mittelwert und diagonale Varianz. Es gibt keinen über Fenster hinweg erhaltenen Zustand. Standardnormalprior ersetzt den Roboterfortschrittsprior. Training verwendet Gauß-NLL plus gewichteten mittleren KL-Term. Standardrauschen 0; optionale Augmentation ausschließlich im Training. Auswertung verwendet Posterior-Mittelwerte, keine Zufallsstichprobe. Exportiert wird μ_z(X_t), ausgelesen nach Verarbeitung des vollständigen Fensters. Hidden Size 64, eine Schicht, Batch 128, 100 Epochen und KL-Gewicht 1 sind MLTrace-Defaults.

L=36 ist bei **keinem** Modell ein Paperparameter. CPU ist unterstützt; CUDA wird ausschließlich über die bestehende Scheduler-Zuweisung sichtbar gemacht. MPS wird aktuell nicht verwendet. Seed 42 ist ein MLTrace-Default; Bibliotheksversionen und Rechengerät stehen im Ergebnismanifest. Numerisch identische Ergebnisse sind nur innerhalb derselben Laufzeit-/Gerätekonfiguration zu erwarten.

## Checkpoint und Scores

Validation beeinflusst Checkpoint-Auswahl, Test niemals. Ohne Validation wird die letzte abgeschlossene Epoche gewählt. Mit Validation wird unabhängig vom Early-Stopping-Schalter der kleinste Validation-Wert gewählt; bei Gleichstand bleibt die frühere Epoche:

| Modell | Auswahl | Metrik |
| --- | --- | --- |
| USAD | validation_reconstruction_score | Mittlerer Window-Score mit eingefrorenem α und β=1−α |
| TCN-AE | validation_loss | Mittlerer Log-Cosh-Loss |
| LSTM-VAE | validation_loss | Deterministische, rauschfreie Gauß-NLL + gewichteter KL-Term |

Alle Scores, Rekonstruktionen und Repräsentationen entstehen erst nach erneutem Laden dieses Checkpoints. Window-Score ist der mittlere Fehler über L×D, Endpoint-Score nur über die D Sensoren der letzten Position. USAD verwendet α·(X−AE1(X))² + β·(X−AE2(AE1(X)))², TCN-AE MSE, LSTM-VAE Gauß-NLL. Varianz ist mittels Softplus plus 10⁻⁶ positiv abgesichert. Keine Überlappungsmittelung, Glättung, Schwellenwertanpassung oder Klassifikationskennzahlen.

Die sichtbare Rekonstruktion ist stets die **letzte Fensterposition**. Standarddarstellung sind Originaleinheiten, optional Min-Max-skaliert. Das LSTM-Band ist μ ± eine Standardabweichung; USAD zeigt die drei Rekonstruktionspfade getrennt. Scores bleiben unabhängig von der Darstellung im skalierten Raum.

## Speicherung und API

Tabellen: `time_series_models`, `time_series_pipelines`, `time_series_runs`, `time_series_epoch_metrics`. Referenzierte Datenbasen, Splits, Modelle und Pipelines sind löschgeschützt. Aktive Läufe müssen erst abgebrochen werden. Änderungen an Quellen verändern vorhandene Snapshots nicht.

Projektartefakte unter `time_series_runs/<run_id>/`:

- `snapshot.json`: Quellenhash, Split und Tags, Sensorreihenfolge, Skalierung, L/Schritt/Gap-Regel, Herkunft, Architektur einschließlich Pooling und Repräsentationsform, Trainingsparameter, Seed, Auswahlmetrik.
- `input.npz`: eingefrorene Werte, skalierte Werte, UTC-Zeitstempel als int64-Nanosekunden, Gruppen, Intervall-/Sequenzzuordnung und vollständige Fensterendpunkte.
- `weights.pt`: ausgewählter State-Dict; SHA-256 steht im Checkpoint-Vertrag.
- `results-*.npz`: Blöcke bis 1.024 Endpunkte, Window-/Endpoint-Scores, letzte Rekonstruktionen und latente Vektoren; zusätzlich USAD-Pfade bzw. VAE-Varianz.
- `manifest.json`: Blockverzeichnis, Encoderform und Ausleseregel, Runtime, Sensorreihenfolge, Eingabefingerprint und Checkpoint-Identität.

USAD exportiert E(X_t), TCN den zeit-/kanalweise abgeflachten gepoolten Bottleneck, LSTM-VAE den deterministischen vollständigen Fenstermittelwert. `z_sensor_ref` ordnet Endpunkte einem Block und dessen Zeile zu. Der ZIP-Export enthält identische Blocknamen mit `end_indices` und `z_sensor`, eine Zuordnungstabelle und sämtliche Metadaten.

Alle Endpunkte liegen unter `/api/time-series` und verwenden den vorhandenen Projektkontext:

- `GET model-definitions`; `GET/POST models`, `GET/PUT/DELETE models/{id}`.
- `POST pipelines/preview`; `GET/POST pipelines`, `GET/PUT/DELETE pipelines/{id}`; `POST pipelines/{id}/runs`.
- `GET runs`, `GET/DELETE runs/{id}`, `POST runs/{id}/abort`, `GET runs/{id}/logs`.
- `GET runs/{id}/series`: subset, sensor, scaled, offset, limit, optional start/end.
- `GET runs/{id}/representations`: subset, offset, limit; latente Vektoren plus Endpunktvertrag.
- `GET runs/{id}/windows/{endpoint_index}`: deterministische Rekonstruktion des vollständigen eingefrorenen Fensters aus gespeicherten Gewichten.
- `GET runs/{id}/export.csv?sensor=0&scaled=false`: alle Gruppen und Scores/Rekonstruktionen des ausgewählten Sensors.
- `GET runs/{id}/representations.zip`: alle Gruppen und latenten Vektoren mit Metadaten.

Die spätere Bild-Sensor-Fusion kann diese Repräsentationen anhand der Endzeitpunkte verknüpfen; ein Fusionsmodell ist nicht Bestandteil dieser Erweiterung.
## Dezimalformate in Sensorwerten

Sensorspalten unterstützen Dezimalpunkt und Dezimalkomma, etwa `65.67` beziehungsweise `65,67`, auch innerhalb derselben Spalte. Bei einer kommagetrennten CSV müssen Dezimalkomma-Werte wie üblich als CSV-Felder in Anführungszeichen stehen (`"65,67"`). Semikolon, Tab und Pipe als Feldtrenner bleiben ebenfalls unterstützt. Tausendertrennzeichen werden nicht automatisch interpretiert; leere, nichtendliche und fehlerhafte Werte bleiben Fehler mit Angabe von Sensor, Zeitpunkt und Originalwert.

Die Umwandlung erfolgt ausschließlich für Sensorspalten bei Preview und Training. Original-CSV, Zeitstempel und Labels bleiben unverändert. Bereits importierte Datenbasen müssen nicht neu hochgeladen werden; die Pipeline-Vorschau kann erneut geprüft werden.
