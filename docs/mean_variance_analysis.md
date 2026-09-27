# Mittelwert-/Varianzvergleich

Der Reiter im Arbeitsbereich **Bilddaten** vergleicht zwei Zeiträume desselben
Train/Test-Datensatzes mit derselben Preprocessing-Pipeline. Er erzeugt zwei
statische, farbige PNG-Heatmaps mit Legende, Zeiträumen und Bildanzahlen.

## Ablauf

1. Train/Test-Datensatz und Preprocessing auswählen (gleich große Graustufenbilder).
2. Normalphase und Anomaliephase festlegen; Beginn und Ende sind einschließlich.
   Die gemeinsame Bibliothek [Gespeicherte Zeiträume](time_range_presets.md) lässt
   sich in beiden Feldern verwenden. Sie übernimmt nur Beginn und Ende.
3. Sampling wählen. In der Normalphase sind regelmäßiges Sampling oder eine
   Zufallsanzahl ohne Zurücklegen mit Seed möglich; in der Anomaliephase nur
   regelmäßiges Sampling. Datensatzsampling gilt zuerst, anschließend werden
   deduplizierte, chronologisch nach Zeitpunkt und Dateipfad geordnete Bilder
   innerhalb des jeweiligen Zeitraums an den Positionen n, 2n, 3n … ausgewählt.
4. Für jede Heatmap unabhängig eine automatische oder manuelle Skala festlegen,
   **Auswahl prüfen**, dann **Berechnung starten**.
5. Ergebnisse öffnen und PNGs herunterladen. Gespeicherte Läufe können als Vorlage
   übernommen, abgebrochen oder gelöscht werden. Änderungen erzeugen neue Läufe.

Standards sind Sampling 1, Seed 42 und automatische Skalen. Überlappende Zeiträume
und einzelne Zeitpunkte sind erlaubt. Übernehmen einer Zeitraumvorlage und jede
Formularänderung verwerfen die Auswahlvorschau. Zeitlücken werden nicht aufgefüllt.

## Rechnung und Interpretation

Pro Pixel werden Mittelwert und zeitliche Populationsvarianz berechnet:

```text
mean_normal = mean(Normalbilder)
mean_anomalie = mean(Anomaliebilder)
var_normal = mean((Normalbilder - mean_normal)²)
var_anomalie = mean((Anomaliebilder - mean_anomalie)²)

mean_difference = abs(mean_normal - mean_anomalie)
variance_difference = var_anomalie - var_normal
```

Die Berechnung erfolgt in float64 mit Welfords Online-Verfahren und `ddof=0`,
ohne zusätzliche Normalisierung, Shift oder Clipping. Der Arbeitsspeicherbedarf
ist unabhängig von der Anzahl Bilder. Bei einem einzelnen Bild ist die Varianz
null; Vorschau und Ergebnis weisen auf die fehlende zeitliche Vergleichsbasis hin.

- **Absoluter Mittelwertunterschied:** Viridis-Farbskala ab null; Einheit der
  Pipeline-Ausgabe. Automatische Obergrenze ist die größte Differenz.
- **Varianzdifferenz:** Blau für geringere Varianz in der Anomaliephase, Weiß bei
  null, Rot für höhere Varianz. Die Skala ist symmetrisch; automatisch gilt die
  größte absolute Varianzdifferenz. Einheit ist das Quadrat der Pipeline-Einheit.

Manuelle positive Grenzen begrenzen nur die Farbdarstellung. Die Ergebnisdaten
enthalten weiterhin die tatsächlichen Minima und Maxima. Nullbilder erscheinen
einheitlich in der Nullfarbe und tragen einen Hinweis. In diesem Fall verwendet
die automatische Darstellung intern eine Ersatzspanne zur Vermeidung einer
Division durch null; die gespeicherte tatsächliche Grenze bleibt null.

Bildorientierung (Ursprung oben links) und Seitenverhältnis bleiben erhalten,
ohne glättende Interpolation. Beide PNGs enthalten Titel, Farbskala, Zeiträume und
Bildanzahlen. Es gibt keine Rohdaten-Downloads, Ausgangskarten oder Videoausgabe.

## Persistenz, Jobs und Schnittstellen

Migration `0063_mean_variance` folgt auf `0062_time_range_presets` und ergänzt
`mean_variance_runs`. Jeder Lauf speichert Konfiguration, Datensatzbeschreibung,
Pipeline-Snapshot und ausgewählte Dateien mit Zeitstempeln, Größe und Änderungszeit.
Änderungen an den Quellen seit dem Start oder während des Lesens führen zum Fehler.
Gespeicherte Ergebnisse sind unabhängig von späteren Änderungen an Pipeline,
Datensatzregeln oder Zeitraumvorlagen.

`mean_variance` ist ein CPU-Auftrag im bestehenden Scheduler. Warteschlange,
Projektübersicht, Abbruch, Protokoll und Laufdetails sind integriert. Im Data Manager
erscheinen die Läufe mit ihren Datensatzabhängigkeiten; Löschung bereinigt die
Artefakte unter `mean_variance_runs/<id>/`. Die Pipeline ist eingefroren und wird
nicht als dauerhafte Fremdschlüssel-Abhängigkeit gehalten.

Die projektgebundene API verwendet `/api/mean-variance-analysis`:

| Methode | Pfad | Zweck |
| --- | --- | --- |
| POST | `/preview` | Bildanzahlen und Auswahlfehler |
| POST / GET | `/runs` | Lauf erstellen / auflisten |
| GET / DELETE | `/runs/{id}` | Lauf lesen / löschen |
| POST | `/runs/{id}/abort` | Abbruch anfordern |
| GET | `/runs/{id}/log` | Letzte Protokolleinträge |
| GET | `/runs/{id}/results` | Zusammenfassung und Heatmap-Metadaten |
| GET | `/runs/{id}/artifacts/{name}` | PNG anzeigen oder mit `download=true` herunterladen |

Konfiguration: `training_dataset_id`, `preprocessing_pipeline_id`, `reference`
(Normalphase), `anomaly`, `mean_scale` und `variance_scale`. Die Zeiträume entsprechen
den Auswahlstrukturen der Referenzbild-Analyse. Jede Skala hat `mode: auto|manual`
und `limit: number|null`; bei manuell muss der Grenzwert positiv und endlich sein.

Ergebnisse enthalten Bildanzahlen, Bildgröße, `ddof: 0`, Warnungen und die Karten
`maps.mean` sowie `maps.variance` mit Dateiname, Einheit, Skalenlimit,
tatsächlichem Minimum/Maximum und Nullbild-Kennzeichen. Als Artefakte sind nur
`mean_difference.png` und `variance_difference.png` freigegeben. Erst wenn beide
Exporte erfolgreich sind und der Lauf den Status `finished` erreicht, sind
Ergebnisse verfügbar. Abbruch oder Fehler veröffentlichen keine Teilergebnisse.

## Verifikation

`backend/tests/test_mean_variance.py` prüft bekannte Pixelwerte, Vorzeichen,
16-Bit-Eingaben, numerische Stabilität, Null- und Einzelbilder, Skalen und
PNG-Metadaten sowie Auswahl, Dateiveränderungen, Fehler, Abbruch, Scheduler,
Data-Manager-Bereinigung, Downloads, Projekttrennung und Migration.
`frontend/src/meanVariance/*.test.ts` prüft Konfiguration, Zeitraumübernahme,
unabhängige Skalen und projektgebundene API-Aufrufe. Gemeinsame Auswahl- und
Dateiprüfungshelfer werden zusätzlich durch die Referenzbild-Regressionstests geprüft.
