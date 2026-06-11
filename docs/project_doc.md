# MLTrace Project Documentation

Stand: 2026-06-10

## Ziel

MLTrace soll eine lokale Plattform werden, mit der Autoencoder-basierte Bildanomalie-Experimente nachvollziehbar verwaltet, gefiltert, verglichen und wiederverwendet werden können.

Das übergeordnete ML-Ziel ist:

- Ein Autoencoder wird auf Normalzustandsbilder trainiert.
- Bei der Inferenz wird der Rekonstruktionsfehler berechnet.
- Bekannte Normalzustände sollen gut rekonstruiert werden.
- Unbekannte oder anomale Zustände sollen schlechter rekonstruiert werden.
- Hoher Rekonstruktionsfehler dient als Anomalieindikator.

Das Plattformziel ist nicht nur Training oder Inferenz auszuführen, sondern die vielen entstehenden Varianten kontrollierbar zu machen: Datenauswahl, Preprocessing, Modellarchitektur, Trainingsparameter, Inferenzmethoden, ROI-Auswertung, Ergebnisse, Filterung und Vergleich.

## Anforderungen

Aus den bisherigen Anforderungen ergeben sich diese Kernpunkte:

- Die Plattform läuft lokal im Einzelbetrieb.
- Sie darf Bilddaten nicht kopieren; Datensätze bleiben an ihren Originalpfaden.
- Datensätze werden über Root-Pfade hinzugefügt.
- Innerhalb dieser Pfade liegen TIFF-Bilder, typischerweise `.tif` und `.tiff`.
- Timestamps werden aus Dateinamen erkannt und zusätzlich als Rohwert gespeichert.
- Beim Hinzufügen eines Datensatzes soll MLTrace den Pfad scannen und Metadaten speichern:
  - vorhandene Unterordner
  - Bildformat
  - Auflösung
  - erstes und letztes Bild pro Ordner
  - zeitlicher Abstand der Bilder
  - erkanntes Timestamp-Format
- Für Trainingsdatensätze sollen beliebig viele Ordner und beliebig viele Zeitbereiche ausgewählt werden können.
- Ein Trainingsdatensatz darf Zeitbereiche aus mehreren verschiedenen Dataset-Roots enthalten.
- Pro Zeitbereich muss ein Sampling-Stride gesetzt werden können, zum Beispiel jedes Bild oder jedes x-te Bild.
- Gespeicherte Trainingsdatensätze müssen inspizierbar sein:
  - Dataset
  - Root-Pfad
  - Ordner
  - Startzeit
  - Endzeit
  - Stride
  - aktuelle Bildanzahl
- Gespeicherte Trainingsdatensätze müssen gelöscht werden können.
- Der Name eines Trainingsdatensatzes soll automatisch aus frühestem Start-Timestamp und spätestem End-Timestamp vorgeschlagen werden.
- Später sollen Modelle, Preprocessing-Methoden, Trainingsparameter, Inferenzmethoden und Resultate eindeutig verwaltbar sein.
- Später soll nach Eigenschaften gefiltert werden können, etwa Preprocessing, Modellarchitektur, Trainingslänge oder Training Dataset.
- Später soll ein trainiertes Modell anhand dieser Eigenschaften eindeutig ausgewählt und für Inferenz verwendet werden können.
- Später sollen mehrere Modelle auf gleicher Inferenzbasis direkt vergleichbar sein.
- Später sollen interaktive Plots für Auswertung und Vergleich vorhanden sein.

## Aktuelle Umsetzung

Die aktuelle Umsetzung enthält Datenkatalog, Trainingsdatensatz-Builder, Preprocessing-Pipelines und eine Methods-Registry.

Technischer Stack:

- Backend: FastAPI, SQLAlchemy, Alembic, Pillow, OpenCV, NumPy
- Datenbank: SQLite lokal unter `.mltrace/mltrace.db`
- Frontend: React, Vite, TypeScript, Mantine
- Betrieb: ohne Docker
- UI-Sprache: Englisch

Aktuell implementierte Seiten:

- `Datasets`
  - Dataset-Root-Pfad hinzufügen
  - Timestamp-Format aus TIFF-Dateinamen erkennen
  - Timestamp-Regex und Python-Datetime-Format bestätigen oder korrigieren
  - Dataset scannen
  - Folder-Summaries anzeigen

- `Training Datasets`
  - Trainingsdatensatz aus gescannten Ordnern erstellen
  - Ranges aus mehreren Dataset-Roots kombinieren
  - Start/Ende pro Range innerhalb der vorhandenen Folder-Zeitgrenzen auswählen
  - Stride pro Range setzen
  - Preview der Bildanzahl anzeigen
  - Trainingsdatensatz speichern
  - gespeicherte Trainingsdatensätze inspizieren
  - gespeicherte Trainingsdatensätze löschen

- `Preprocessing`
  - globale Preprocessing-Pipelines anlegen
  - lineare Pipeline als Graph aus Nodes und Edges speichern
  - Bausteine aus automatisch entdeckten Python-Step-Klassen laden
  - neue Bausteine als eigene Datei unter `backend/app/preprocessing/steps/` ergänzen
  - ein Standardformat über `BasePreprocessingStep` verwenden
  - mitgelieferte Bausteine: `load_image`, `warp_perspective`, `resize`, `crop`, `grayscale`, `normalize_for_preview`, `gaussian_blur`
  - Schritt-für-Schritt-Anleitung zum Erstellen neuer Module siehe [preprocessing_modules.md](preprocessing_modules.md)
  - lineare Bausteinansicht statt freier Graph-Canvas nutzen
  - Bausteine per UI hinzufügen, umsortieren und entfernen
  - Input-/Output-Informationen pro Baustein anzeigen, inklusive Formattyp, Größe, Kanalzahl und dtype nach Preview
  - ein Dataset-Folder als Preview-Quelle wählen
  - erstes Bild des Folders sofort bei Auswahl laden und persistent für alle Bausteine bereithalten
  - Zwischenbild nach jedem Pipeline-Schritt anzeigen
  - Bausteine inline im Baustein selbst konfigurieren (Configure klappt im Block auf, keine separate Panel-Spalte)
  - pro Baustein generisch ein Input-/Output-Vorschaufenster anzeigen; das Eingangsbild ist die Ausgabe des Vorgängerschritts
  - Vorschau automatisch (debounced) aktualisieren, sobald Parameter geändert werden
  - interaktive Bild-Werkzeuge schema-getrieben über `config_schema.ui_control` und eine Frontend-Control-Registry einbinden
  - bei `warp_perspective` (`ui_control: point_picker`) vier Quellpunkte automatisch initialisieren, auf dem Eingangsbild verschieben, als Polygon verbinden und die transformierte Fläche halbtransparent orange einfärben sowie das transformierte Bild daneben prüfen
  - bei `crop` (`ui_control: crop_box`) den Crop-Bereich interaktiv über ein verschieb- und skalierbares, halbtransparent orange eingefärbtes Rechteck auf dem Eingangsbild bestimmen und das beschnittene Bild daneben prüfen
  - bei `crop` die Ausgabegröße wählen: nur croppen, auf die Eingangsgröße oder auf die Pipeline-Ursprungsgröße interpolieren
  - Größen-Parameter neuer Bausteine aus der tatsächlichen Pixelgröße des Vorgängerschritts vorbelegen (`default_from`)
  - die Typ-Kette prüfen: jeder Baustein deklariert per `output_spec` einen I/O-Vertrag (Kanäle/Größe); inkompatible Ketten werden beim Speichern und in der Vorschau hart blockiert
  - Konfigurationswerte gegen `config_schema` validieren (Typ, min/max, enum) und Fehler im UI anzeigen
  - gespeicherte Pipelines laden, bearbeiten (Update) oder als neue Pipeline speichern; Pipeline-Namen sind eindeutig (case-insensitive)
  - gespeicherte Pipelines laden und löschen

- `Methods`
  - globale, benannte Method-Konfigurationen anlegen
  - Method-Definitionen automatisch aus Python-Klassen unter `backend/app/modeling/architectures/` laden
  - ein Standardformat über `BaseMethodDefinition` verwenden
  - initiale Methods: `cnn_autoencoder`, `cnn_vae`, `mean_image`
  - PatchCore, Optical Flow und ähnliche Verfahren sind noch nicht implementiert, können aber später als weitere Method-Klassen ergänzt werden
  - CNN-AE und CNN-VAE über einen sequenziellen Builder definieren: Input, Encoder, Latent, Decoder, Output
  - V1-Layer aus einem kuratierten Layer-Katalog wählen: `Conv2d`, `ConvTranspose2d`, `BatchNorm2d`, `MaxPool2d`, `Upsample`, `Dropout2d`, `Flatten`, `Unflatten`, `Linear`, `ReLU`, `LeakyReLU`, `GELU`, `Sigmoid`, `Tanh`
  - Layer konfigurieren, umsortieren und entfernen
  - Mean Image als schema-getriebenes Formular speichern, ohne Layer-Builder
  - pro Method ein strukturelles Diagramm als JSON erzeugen und im UI anzeigen
  - Method-Konfigurationen laden, aktualisieren, als neue Konfiguration speichern und löschen
  - Method-Namen sind eindeutig (case-insensitive)
  - Verfahren werden über `method_family`, `training_mode`, `artifact_kind` und `supports_training_pipeline` unterschieden
  - CNN-AE und CNN-VAE haben `training_mode=gradient` und erzeugen später Gewichts-Artefakte
  - Mean Image hat `training_mode=fit`, ist ebenfalls ein späterer Training-/Fit-Schritt und erzeugt ein Mean-Image-Artefakt
  - CNN-Methoden werden statisch über Tensor-Shape-Propagation validiert; harte Architekturfehler blockieren Speichern
  - falls Torch/Torchvision installiert sind, kann zusätzlich ein CPU-Dummy-Forward-Check laufen; ohne Torch bleibt die App nutzbar

Aktuelle Datenbankobjekte:

- `datasets`
  - Name, Root-Pfad, Scan-Status, Timestamp-Parser, Scan-Zusammenfassung
- `dataset_folders`
  - Unterordner, Bildanzahl, Start/Ende, Auflösung, Format- und Cadence-Summary
- `dataset_images`
  - Bildpfad, Ordner, Dateiname, Auflösung, Timestamp-Rohwert, geparster Timestamp
- `training_datasets`
  - Name, Notizen, Erstellzeit
- `training_dataset_rules`
  - Folder, Start, Ende, Stride
- `preprocessing_pipelines`
  - Name, Beschreibung, Pipeline-Graph, Erstell- und Änderungszeit
- `method_configurations`
  - Name, Beschreibung, Method-Typ, Method-Family, Training Mode, Artifact-Kind, Builder-Typ, Method-Graph, Method-, Trainings- und Inferenzconfig, Diagramm, Validierung
- `method_configuration_parameters`
  - flach indexierte skalare Parameterpfade für spätere Filterung

Wichtige bewusste Einschränkungen:

- Trainingsdatensätze speichern aktuell Regeln, kein unveränderliches Bildmanifest.
- Wenn sich Dateien in den Originalordnern ändern, können sich aktuelle Bildzahlen ändern.
- Training, Inferenz, ROI-Auswertung und Modellvergleich sind noch nicht implementiert.
- Preprocessing-Pipelines werden noch nicht batchweise auf Trainingsdatensätze angewendet.
- Preprocessing-Pipelines erzeugen noch keine persistenten Bildartefakte.
- Gespeicherte Method-Konfigurationen erzeugen noch keine persistenten Torch-Module, Artefakte oder Trainingsläufe.
- Mean Image wird aktuell nur als Baseline-Konfiguration gespeichert; das eigentliche Mean-Image-Artefakt wird später berechnet.
- Es gibt noch keine Benutzerverwaltung, weil Einzelbetrieb gewünscht ist.
- Es gibt noch keine freie visuelle DAG-Oberfläche; die V1-UI nutzt bewusst eine klare lineare Bausteinansicht, speichert intern aber weiterhin Nodes und Edges.

## Nächster Empfohlener Schritt

Der nächste sinnvolle Schritt ist aus meiner Sicht eine **Training Pipeline / Training Run Definition**, noch bevor echte Trainingslogik tief integriert wird.

Begründung:

Die zentrale Schwierigkeit deines Projekts ist nicht nur, eine Method auszuführen oder ein neuronales Modell zu trainieren, sondern später eindeutig zu wissen:

- welcher Trainingsdatensatz verwendet wurde
- welches Preprocessing verwendet wurde
- welche gespeicherte Method-Konfiguration verwendet wurde
- welche Trainingsparameter verwendet wurden
- wo die Artefakte liegen
- welcher Run erfolgreich war
- welche trainierte Method beziehungsweise welches erzeugte Artefakt später für Inferenz und Vergleich ausgewählt werden soll

Nachdem Training Datasets, Preprocessing Pipelines und Methods jetzt als wiederverwendbare Bausteine existieren, sollte der nächste Schritt diese Bausteine zu einer speicherbaren Training Pipeline verbinden. Dabei sollen nur Methods auswählbar sein, deren gespeicherte Konfiguration `supports_training_pipeline=true` hat.

## Vorgeschlagener Nächster Implementationsschnitt

Der nächste Implementationsschnitt sollte folgende Funktionen enthalten:

- Seite `Training Pipelines`
  - neue Training Pipeline anlegen
  - gespeicherten Trainingsdatensatz auswählen
  - gespeicherte Preprocessing Pipeline auswählen
  - gespeicherte Method-Konfiguration auswählen
  - Trainingsparameter aus der Method-Konfiguration übernehmen und überschreibbar machen
  - Pipeline speichern
  - Pipelines tabellarisch anzeigen
  - nach Dataset, Zeitbereich, Preprocessing, Method-Typ, Method-Family und Parametern filtern

- Backend-Modelle:
  - `training_pipelines`
  - `training_pipeline_datasets`
  - `training_runs`
  - `training_run_artifacts`

- V1-kompatible einfache Plugin-Struktur:
  - Python-Module registrieren Method- und Preprocessing-Optionen
  - Konfigurationen werden als JSON gespeichert
  - noch kein visueller DAG

- Run-Status:
  - `draft`
  - `queued`
  - `running`
  - `succeeded`
  - `failed`

- Artefakt-Tracking:
  - Artefaktpfad
  - Config-Snapshot
  - Log-Pfad
  - Trainingsmetriken

Wichtig: In diesem Schritt kann Training zunächst noch manuell oder als Dummy-Runner ausgeführt werden. Entscheidend ist, dass die Plattform die Pipeline und ihre Konfiguration korrekt abbildet. Danach kann ein echter PyTorch-Runner angeschlossen werden.

## Roadmap

Empfohlene Reihenfolge:

1. Dataset Catalog und Training Dataset Builder stabilisieren.
2. Preprocessing-Pipeline-Builder stabilisieren.
3. Method Registry und Method Builder stabilisieren.
4. Training Pipeline / Training Run Definition mit Method-, Preprocessing- und Trainingsconfig bauen.
5. Lokalen PyTorch-Runner integrieren.
6. Method-Artefakte und Metriken speichern.
7. Inferenz-Run-Registry hinzufügen.
8. ROI-Konfigurationen und Rekonstruktionsfehler-Auswertung hinzufügen.
9. Interaktive Ergebnisplots und Modellvergleich bauen.
10. Plugin-System erweitern.
11. Später optional visuellen Pipeline-/DAG-Builder auf denselben Plugins aufsetzen.

## Akzeptanzkriterien Für Den Nächsten Schritt

Ein nächster Schritt gilt als erfolgreich, wenn:

- ein Training Run ohne Training-Ausführung gespeichert werden kann
- der Run eindeutig auf einen gespeicherten Trainingsdatensatz verweist
- Method-, Preprocessing- und Trainingsparameter strukturiert gespeichert werden
- mehrere Runs mit unterschiedlichen Konfigurationen angelegt werden können
- Runs in einer Tabelle sichtbar und filterbar sind
- die gespeicherte Konfiguration später für echte Training-Ausführung verwendet werden kann
