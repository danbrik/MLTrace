# Trainingsparameter und separate Validierung

Der vierte Schritt in Training Pipelines gruppiert die tatsächlich wirksamen Parameter nach Trainingsablauf, Optimizer, Loss, Validierung, Early Stopping und Berechnung/Datenladen. SSIM-Felder erscheinen nur bei einem aktiven SSIM-Loss; Prediction-Felder hängen zusätzlich vom Modellzweig und Trainingsziel ab. AMP wirkt nur bei CUDA. Prefetch benötigt mindestens einen Worker.

Neue Pipelines starten ohne Validierung. Mit **Validierung verwenden** werden ein oder mehrere vorhandene Train/Test-Datensätze ausgewählt. Deren Regeln und Sampling gelten unverändert; es wird kein zusätzlicher Anteil abgetrennt. Training und Validierung nutzen dieselbe Vorverarbeitung. Beide Shuffle-Schalter steuern unabhängig die Reihenfolge, bei Sequenzen ausschließlich die Reihenfolge vollständiger Clips. Überschneidungen aufgelöster Quelldateien einschließlich Zielbildern sind nicht zulässig. Der Dummy Test prüft die ausgewählten Quellen und Ausgabegrößen und zeigt Bild- beziehungsweise Clipzahlen getrennt an.

Validierung verwendet den aktiven Trainingsverlust ohne Gradienten. Der Verlust wird nach tatsächlicher Batchgröße gewichtet. Ein vollständig unlesbarer Validierungsdatensatz führt zu einem Fehler. Early Stopping überwacht mit Validierung deren Verlust, sonst den Trainingsverlust. Direkt angepasste Verfahren und f-AnoGAN erhalten keinen zusätzlichen Validierungsdurchlauf.

Gespeicherte Pipelines werden schreibgeschützt geöffnet. Alte Pipelines und Checkpoints bleiben im Modus `legacy_fraction` und behalten ihr bisheriges Verhalten. Beim Bearbeiten muss ausdrücklich keine oder separate Validierung gewählt werden. Bereits vorhandene Ergebnisse werden nicht verändert.

## Schnittstellen

Die bestehenden Training-Pipeline-Endpunkte erhalten `validation_mode` (`none`, `external`, `legacy_fraction`), geordnete `validation_dataset_ids` und `validation_shuffle`. Neue Clients senden den Modus ausdrücklich. Für alte API-Clients bleibt der Vorgabewert `legacy_fraction`. Antworten liefern `validation_datasets` und `total_validation_images`; Dry Runs zusätzlich `training_sample_count`, `validation_sample_count` und `sample_kind`. Laufdaten und Checkpoints halten Modus, Quellen und Shuffle fest; Validierungsquellen werden bei Duplikat- und Fortsetzenprüfungen sowie Löschabhängigkeiten berücksichtigt.

Die Migration `0066_training_validation` ergänzt Felder und Verknüpfungen in jeder Projektdatenbank. Die Definitionen unter `modeling/training_ui.py` liefern gemeinsame `ui_groups`, `visible_if` und `supports_validation`-Metadaten. Frontend und Backend werten dieselben Bedingungen aus. Nur aktive Parameter werden für neue Konfigurationen validiert und gespeichert.
