# Varianzvergleich

Der Reiter **Varianzvergleich** im Arbeitsbereich Bilddaten vergleicht ein bis
sechs Paare aus Normalzustand und Anomaliephase. Datensatz, Preprocessing-Pipeline
und Sampling sind für sämtliche Zeiträume gemeinsam. Die Ausgabe ist ein
zusammenhängender PNG-Plot mit drei Spalten und einer Zeile je Paar.

## Bedienung

1. **Neuer Vergleich** öffnet einen Entwurf mit einem leeren Paar, Sampling 1 und
   automatischen Farbskalen. Train/Test-Datensatz und Graustufen-Pipeline auswählen.
2. Normalzustand und Anomaliephase für u1 festlegen; bis zu fünf weitere Paare
   hinzufügen. Die Reihenfolge ergibt u1 bis u6. Nur vollständige Paare sind gültig.
   Jeder Zeitraum unterstützt die gemeinsame Bibliothek
   [Gespeicherte Zeiträume](time_range_presets.md) samt Speichern, Bearbeiten und Löschen.
3. Beginn und Ende zählen einschließlich. Einzelzeitpunkte und überlappende
   Zeiträume sind erlaubt. Datensatzsampling gilt zuerst, dann Deduplizierung und
   stabile Sortierung nach Aufnahmezeitpunkt und Pfad. Das zusätzliche Sampling
   wählt je Zeitraum n, 2n, 3n …; Restblöcke entfallen. Es gibt kein Zufallssampling.
4. Varianz- und Differenzskala unabhängig automatisch oder manuell festlegen.
   **Auswahl prüfen** zeigt je Paar/Rolle verfügbare, ausgewählte und übrige Bilder.
   Jede Änderung verwirft diese Vorschau. Erst eine gültige Vorschau erlaubt den Start.
5. **Lauf öffnen** zeigt die gespeicherten Einstellungen schreibgeschützt zusammen
   mit dem passenden Ergebnis. **Als Vorlage übernehmen** erstellt einen bearbeitbaren
   Entwurf für einen neuen Lauf. Laufwechsel und Polling verändern keinen Entwurf.

Die gespeicherten Datensatz- und Pipelinebeschreibungen bleiben auch bei späteren
Änderungen sichtbar. Für einen neuen Lauf müssen die ausgewählten Ressourcen
weiterhin verfügbar sein. Projektwechsel setzen die gesamte Seite zurück.

## Berechnung und Darstellung

Pro Pixel und Paar gilt:

```text
var_normal   = mean((Normalbilder − mean_normal)²)
var_anomalie = mean((Anomaliebilder − mean_anomalie)²)
differenz    = var_anomalie − var_normal
```

Float64, Welfords Online-Verfahren und Populationsvarianz (`ddof=0`) bleiben
unverändert. Mittelwerte sind nur interne Hilfsgrößen; es gibt keine zusätzlichen
Normalisierungen, Shift- oder Clipping-Schritte. Ein einzelnes Bild ergibt Varianz
null mit Hinweis. Alle Ausgaben müssen gleich große, endliche Graustufenbilder sein.
Bilder werden sequenziell und Paare nacheinander verarbeitet. Karten werden vor
Rendering temporär auf Platte abgelegt; keine vollständige Bildserie wird im
Arbeitsspeicher gehalten. Temporäre Dateien werden auch bei Fehlern bereinigt.

- Spalten: **Normalzustand**, **Unruhe**, **Differenz**; Zeilen: **u1 … u6**.
- Eine Weiß–Rot-Skala ab null gilt für alle Normal-/Anomaliekarten. Die zweite,
  symmetrische Blau–Weiß–Rot-Skala gilt für alle Differenzkarten.
- Automatische Grenzen sind die größte Varianz beziehungsweise größte absolute
  Differenz über sämtliche Paare. Manuelle positive Grenzen beeinflussen nur die
  Farbdarstellung, nicht die Berechnung oder gespeicherten Minima/Maxima.
- Zwei horizontale Farblegenden: **Variance (gray value²)** und
  **Variance difference (gray value²)**. Die Einheit wurde bewusst als Grauwert²
  festgelegt; es findet keine physikalische Kalibrierung statt.
- **x (Pixel)** und **y (Pixel)** stehen jeweils einmal außen. Zahlenmarkierungen
  stehen nur links beziehungsweise unten. Ursprung oben links, gleiches
  Seitenverhältnis und keine glättende Interpolation.
- Die Abbildung wächst mit der Paaranzahl. Nullfälle haben ihre Nullfarbe und
  Hinweise; eine entartete automatische Skala zeigt nur die Nullmarkierung.

Zeiträume, Bildanzahlen, Datensatz und Pipeline stehen außerhalb des Plots in der
Oberfläche und in den PNG-Metadaten. Keine Ausgangsbilder, Mittelwertkarten,
Rohdaten-Downloads oder Videos werden für neue Läufe angeboten.

## Formate, Persistenz und Kompatibilität

Modell, Tabellenname, Scheduler-Kennung `mean_variance` und API-Präfix
`/api/mean-variance-analysis` bleiben stabil. Keine neue Migration ist erforderlich.
Die Konfiguration neuer Läufe enthält:

```json
{
  "version": 2,
  "training_dataset_id": 1,
  "preprocessing_pipeline_id": 1,
  "sampling_rate": 1,
  "pairs": [{
    "normal": {"start": "2026-01-01T10:00:00", "end": "2026-01-01T11:00:00"},
    "anomaly": {"start": "2026-01-01T12:00:00", "end": "2026-01-01T13:00:00"}
  }],
  "variance_scale": {"mode": "auto", "limit": null},
  "difference_scale": {"mode": "auto", "limit": null}
}
```

`POST /preview` gibt für Version 2 `pairs` mit den bisherigen Auswahlzahlen
`reference`/`anomaly` und eine gemeinsame Fehlerliste mit Paarbezeichnungen zurück.
`POST/GET /runs`, `GET/DELETE /runs/{id}`, `POST /runs/{id}/abort`,
`GET /runs/{id}/log` und `GET /runs/{id}/results` behalten ihre Aufgaben.
Alle Anfragen bleiben projektgebunden.

Neue Ergebnisse enthalten `version: 2`, `filename: variance_comparison.png`,
Paarzeiträume, Bildanzahlen, Warnungen, Minima/Maxima/Nullkennzeichen je Karte sowie
beide gemeinsamen Skalenlimits. `GET /runs/{id}/artifacts/variance_comparison.png`
zeigt den Plot; `?download=true` lädt ihn herunter. Dateipfade temporärer Karten
werden nicht veröffentlicht. Downloads sind erst nach vollständigem Export und
Status `finished` verfügbar.

Alte Konfigurationen ohne Version bleiben lesbar und ausführbar; bestehende
API-Clients mit dem alten Format bleiben kompatibel. Vorhandene Exporte werden
nicht geändert oder neu berechnet. Die Oberfläche zeigt bei alten Läufen nur die
bisherige Varianzdifferenz. Bei Übernahme als Vorlage entsteht ein Paar. Ein
abweichendes oder zufälliges altes Sampling wird nicht stillschweigend umgerechnet:
Der Nutzer muss vor der Vorschau ein gemeinsames n auswählen. Die alte Varianzskala
wird zur neuen Differenzskala; die neue Varianzskala beginnt automatisch.

Dateiidentitäten und Pipeline werden beim Start eingefroren. Fehlende oder
veränderte Quellen führen zum Fehler. CPU-Scheduler, Fortschritt über alle Paare,
Abbruch, Protokoll, Projektverwaltung und Data-Manager-Löschung einschließlich
Artefakten unter `mean_variance_runs/<id>/` bleiben integriert.

## Prüfung

`test_variance_pairs.py` prüft Version 2, ein/sechs Paare, gemeinsame Auswahl,
Vorzeichen, Nullfälle, Metadaten, globale Skalen, Beschriftungen, Fehler, Abbruch,
Download und Bereinigung. `test_mean_variance.py` erhält die Altformat-Regression;
`test_reference_image.py` prüft die gemeinsam verwendete Bildauswahl.
Frontendtests prüfen Validierung, globale Einstellungen, Kopieren alter/neuer
Konfigurationen, erforderliche Sampling-Auswahl und projektgebundene API-Aufrufe.

## ROI-Auswertung

Nach einem fertigen Lauf öffnet der Unterreiter **ROI-Auswertung** eine gemeinsame,
achsenparallele ROI für alle Paare. Der Editor zeigt je Paar das Normalmittelbild
und die vorzeichenbehaftete Varianzdifferenz. Vier Eckpunkte und das Rechteck sind
ziehbar; fokussierte Punkte lassen sich mit Pfeiltasten um einen Pixel, mit
Umschalt um zehn Pixel bewegen. Die Anfangsauswahl umfasst das ganze Bild, die
Heatmap-Deckkraft beträgt 50 %. Der Paarwechsel verändert die gemeinsame ROI
nicht. Koordinaten bezeichnen Pixelgrenzen: `x, y, width, height` entspricht exakt
`image[y:y+height, x:x+width]`.

**Fertig** erstellt einen CPU-Auftrag für zwei separate PNG-Downloads: den Plot
mit Gesamtbild/ROI-Rahmen und Ausschnitt je u1…u6 sowie eine Ergebnistabelle.
Beide Ansichten verwenden dieselben Hintergrund- und Heatmap-Pixel, globale
Graustufengrenzen, die gespeicherte Differenzskala und die gewählte Deckkraft.
Der Ausschnitt wird weder entzerrt noch geglättet. Gemeinsame äußere Pixelachsen
und die Legende `Variance difference (gray value²)` beschriften den Plot.
Eckkoordinaten und Ausschnittgröße stehen nur in der Oberfläche und in den
PNG-Metadaten. Metadaten enthalten außerdem die ursprüngliche Konfiguration,
Datensatz, Pipeline, Zeiträume und Bildanzahlen.

Die Tabelle misst **positive Varianzzunahme**, nicht die vorzeichenbehaftete
Nettosumme: `Z = maximum(var_anomalie - var_normal, 0)` und
`Anteil = sum(Z[roi]) / sum(Z)`. Der Flächenanteil ist
`width * height / (Gesamtbreite * Gesamthöhe)`. Negative Werte bleiben in der
Heatmap sichtbar, gehen aber nicht in diese Kennzahl ein. `Mean` ist das
ungewichtete Mittel der gültigen paarweisen Anteile. Ohne positive Zunahme ist
der Anteil undefiniert (`—`); diese Paare werden unter Angabe der gültigen
Anzahl vom Mittelwert ausgeschlossen. Grenzen und Transparenz beeinflussen
niemals die Kennzahlen. Es gibt keinen CSV- oder numerischen Rohdaten-Download.

Neue Läufe bewahren pro Paar Normalmittelbild und Differenzkarte in `float64`
auf. Für alte V1/V2-Läufe bietet **ROI-Daten nachberechnen** eine explizite
Nachbereitung anhand des eingefrorenen Manifests und Pipelinegraphen, mit
unveränderten Dateiidentitäts- und Größenprüfungen. Geänderte oder fehlende
Dateien verursachen einen Fehler; Status, Konfiguration und ursprüngliche PNGs
des Elternlaufs bleiben erhalten.

Die Migration `0064_variance_roi` ergänzt projektlokale `VarianceRoiJob`-Datensätze.
CPU-Aufträge der Scheduler-Art `variance_roi` führen `prepare` oder `evaluate`
aus. Pro Elternlauf darf genau ein Auftrag aktiv sein; ein partieller eindeutiger
Index sichert dies auch bei gleichzeitigen Anfragen. Exporte sind unveränderliche
Revisionen. Öffnen zeigt die zuletzt erfolgreich abgeschlossene Auswertung samt
ROI und Deckkraft. Abbruch und Fehler einer neuen Revision lassen das vorherige
Ergebnis verfügbar. Beim Entfernen eines fertigen Nachbereitungsauftrags bleiben
die numerischen Grundlagen beim Elternlauf. Aktive ROI-Aufträge blockieren das
Löschen des Elternlaufs; Data-Manager-Kaskaden löschen Kinder zuerst und bereinigen
alle Artefakte.

### API

Alle Endpunkte unter `/api/mean-variance-analysis` verwenden die bestehende
Projektbindung (`X-MLTrace-Project-ID`, für Bilder alternativ `project_id`).

- `GET /runs/{id}/roi`: Verfügbarkeit, Bild-/Paarbeschreibung, letzter Auftrag und
  zuletzt erfolgreich gespeicherte Auswertung.
- `POST /runs/{id}/roi/prepare`: Fehlende numerische Grundlagen nachberechnen.
- `POST /runs/{id}/roi/evaluate`: Export mit
  `{"roi":{"x":0,"y":0,"width":100,"height":80},"opacity":0.5}` starten.
- `GET /runs/{id}/roi/images/{pair}/{layer}`: Nullbasierter Paarindex;
  `layer` ist `background` oder `heatmap`.
- `GET /runs/{id}/roi/artifacts/{job_id}/{name}`: Fertige `roi_comparison.png`
  beziehungsweise `roi_table.png`; `download=true` erzwingt Download.
- `GET /roi-jobs`, `GET /roi-jobs/{id}/log`, `POST /roi-jobs/{id}/abort`,
  `DELETE /roi-jobs/{id}`: Scheduler- und Verwaltungsfunktionen.

Ein PNG ist erst abrufbar, wenn seine Exportrevision vollständig abgeschlossen
ist. Numerische Karten und interne JSON-Dateien sind über diese Artefakt-API
nicht abrufbar.
