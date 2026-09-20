# DINOv3-Repräsentationsanalyse

Der projektbezogene Bereich untersucht die Struktur eingefrorener DINOv3-CLS-Features.
Normal, Anomalie und Puffer sind drei gleichberechtigte Ground-Truth-Labels.
Sie beeinflussen weder den Encoder noch PCA, UMAP oder KMeans. Ergebnisse sind
explorativ; ein farblich getrenntes 2D-Diagramm ist keine Validierung eines Klassifikators.

## Auswahl

Zuerst einen Train/Test-Datensatz und eine Preprocessing-Pipeline wählen. Deren
Crop/ Warp definiert die gemeinsame ROI. Bereiche mit Beginn, Ende, Label,
Samplingrate und optional Zufall hinzufügen. Nicht hinzugefügte Bereiche sind ausgeschlossen.
Bereiche sind links geschlossen und rechts offen; überlappende Bereiche werden abgelehnt.
Anomaliebereiche erhalten eigene stabile Ereignis-IDs A1, A2 usw.

Der gespeicherte Datensatz-Stride gilt zuerst. Danach werden Dateipfade dedupliziert,
Bilder nach Zeitstempel und Pfad sortiert und pro Bereich in vollständige Blöcke
aufgeteilt. Samplingrate 30 zählt **30 ausgewählte Bilder**, keine Sekunden.
Ohne Zufall wird das letzte Bild jedes Blocks verwendet, mit Zufall eines gleichverteilt
gewählt. Ein Restblock entfällt: 65 Bilder ergeben bei Rate 30 genau zwei Stichproben.
Der Seed wird zusammen mit der stabilen Bereichs-ID verwendet; Umsortieren ändert
keine Stichprobe. Die Vorschau zeigt Bildzahlen, verworfene Reste und die ROI.

## Modell und Installation

Im Backend-Environment `pip install -e '.[ml]'` ausführen. Zusätzlich zu Torch und
Torchvision werden timm ab 1.0.29, huggingface-hub, safetensors, scikit-learn und umap-learn benötigt.
Die übrige Anwendung kann ohne diese optionalen Pakete starten.

Verwendet wird DINOv3 ViT-S/16 mit LVD-1689M-Gewichten aus der öffentlichen
[timm-Veröffentlichung](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m).
**Kein Hugging-Face-Konto, Token oder Login ist erforderlich.** Die Dateien werden
von Hugging Face anonym geladen (`token=False`), auch wenn auf dem Server ein Token
hinterlegt ist. Festgelegt ist die Revision `3bf4720a82ec2066db88137180ff1f83a675cef0`.
Konfiguration, Safetensors-Gewichte und Lizenz werden unter
`.mltrace/model_cache/huggingface` projektübergreifend zwischengespeichert
(optional über `MLTRACE_MODEL_CACHE_DIR` konfigurierbar). Nach dem ersten Download
wird der Cache ohne Netzwerkzugriff verwendet. Die Bilder werden lokal verarbeitet
und nicht hochgeladen. Die Modellarchitektur stammt aus dem installierten timm-Paket;
es wird kein Python-Code aus dem Modellrepository ausgeführt.

Pipeline-Ausgaben werden über den vollständigen Integer-Datentypbereich auf [0,1]
abgebildet. Float-Ausgaben müssen bereits endlich und innerhalb [0,1] sein.
Graustufen werden auf drei identische Kanäle kopiert, Bilder mit bikubischer
Interpolation auf 224×224 skaliert und gemäß Modellprozessor normalisiert. Es gibt
keinen zusätzlichen Crop. Diese feste Skalierung kann das Seitenverhältnis ändern;
bei Bedarf die gewünschte Geometrie bereits in der Pipeline herstellen.

Der Encoder läuft eingefroren im Evaluationsmodus ohne Gradienten. Extrahiert wird
der CLS-Token `forward_features(...)[:, 0, :]`, ohne zusätzliche
Featurestandardisierung oder L2-Normalisierung. Die RoPE-Perioden werden gemäß
timm-Modellbeschreibung auf die bfloat16-Präzision des ursprünglichen Checkpoints
gebracht. Die konkrete Modellrevision, Quelle, Normalisierung und Bibliotheksversionen
werden im Lauf gespeichert. Der Architekturstandard für durchschnittliches Pooling
wird ausdrücklich durch CLS-Pooling ersetzt.

## Berechnungen und Ergebnisse

PCA-2D und UMAP-2D werden auf denselben CLS-Features berechnet. UMAP verwendet eine
euklidische Distanz, `min_dist=0.1`, höchstens 15 Nachbarn, zufällige Initialisierung
und den gespeicherten Seed. Eine separate zentrierte PCA ohne Whitening erhält
standardmäßig mindestens 95 % erklärte Varianz. KMeans nutzt diese Komponenten,
zehn Initialisierungen und standardmäßig k=3; k ist einstellbar.

Vier interaktive Diagramme zeigen beide Projektionen jeweils nach Label oder nach
Anomalie-Ereignis. Normal und Puffer bleiben in der Ereignisansicht eigene Gruppen.
Die Plot-Werkzeugleiste exportiert PNG. ARI, NMI und Cluster×Label-Kreuztabelle
berücksichtigen auch Puffer. Clusternummern bleiben neutrale IDs.

Es müssen mindestens vier Bilder und sowohl Normal- als auch Anomaliebeispiele
vorhanden sein. Ein hinzugefügter Bereich darf nach Sampling nicht leer sein.
Ungültige/konstante Features, zu viele Cluster oder unlesbare bzw. seit dem Einplanen
veränderte Quelldateien brechen den Lauf mit einer Fehlermeldung ab.

## Speicherung und Ausführung

Die API liegt unter `/api/dinov3-analysis`. `/preview` liefert Auswahlzahlen und
Bildvorschau; `/runs` erstellt und listet Läufe. Je Lauf gibt es Status, `/results`,
`/log`, `/abort`, DELETE und `/artifacts/{name}`. Der normale Projektkontext gilt
für sämtliche Zugriffe. Der Jobtyp im Scheduler heißt `dinov3_analysis`.

Die Migration `0057_representation_analysis` ergänzt die Laufverwaltung. Die normale
Projektinitialisierung aktualisiert das Schema. Der Scheduler verwendet CPU oder
die zugewiesene CUDA-GPU. Fortschritt zeigt messbare Bildzahlen sowie Phasennamen;
UMAP/PCA/KMeans zeigen Aktivität statt geschätzter Prozentwerte. Ein Heartbeat läuft
auch während längerer Berechnungen. Abbruch kann bis zur nächsten Unterbrechung
innerhalb einer numerischen Operation dauern.

Konfiguration, Datensatzregeln und Pipeline werden beim Einplanen eingefroren.
Ein Dateimanifest hält die konkrete Stichprobe mit Größe und Änderungszeit fest.
Neue Dateien oder spätere Änderungen an Auswahlregeln beeinflussen diesen Lauf nicht.
Ergebnisse liegen im Projektordner `representation_runs/<id>`:

- `samples.csv`: Bildreferenz, Zeitstempel, Label, Bereich, Ereignis, Projektionen und Cluster.
- `features.npz`: Float32-Featurematrix mit zugehörigen Dateipfaden und Zeitstempeln.
- `analysis.json`: Konfiguration, Snapshots, Metriken und Bibliotheksversionen.
- `manifest.json` und `points.json`: interne Auswahl und Diagrammdaten.

Bestehende Läufe lassen sich erneut öffnen und als editierbare Vorlage übernehmen.
Gleiche Seeds sichern reproduzierbare Zufallsauswahlen; numerische Ergebnisse können
zwischen Hardware- und Bibliotheksversionen geringfügig abweichen.

## Tests

`pytest backend/tests/test_dinov3_analysis.py` verwendet synthetische Bilder und einen
kleinen Ersatzencoder. Die Frontend-Tests liegen in `representation/helpers.test.ts`.
Der optionale echte Modelltest läuft mit
`MLTRACE_TEST_DINOV3=1 pytest backend/tests/test_dinov3_model_integration.py` und benötigt
beim ersten Aufruf eine Netzwerkverbindung für den öffentlichen Download. Er verwendet
ausschließlich synthetische Bilder und prüft auch identische Features nach erneutem
Laden aus dem Cache ohne Netzwerkzugriff.
