# Referenzbild-Analyse

Der Reiter **Referenzbild-Analyse** im Arbeitsbereich Bilddaten vergleicht Bilder
mit einem pixelweise gemittelten Referenzbild. Es ist kein trainiertes Modell
und keine zusätzliche Methodenkonfiguration erforderlich.

## Ablauf

1. Einen Train/Test-Datensatz und eine Preprocessing-Pipeline auswählen. Die
   Pipeline muss gleich große, einkanalige Graustufenbilder liefern.
2. Den Referenzzeitraum einschließlich Beginn und Ende angeben. Entweder jedes
   n-te Bild auswählen oder eine feste Anzahl zufälliger Bilder mit Seed.
3. Den Anomaliezeitraum einschließlich Beginn und Ende und dessen Sampling
   festlegen. Hier gibt es kein Zufallssampling.
4. Kontrast und FPS wählen, **Auswahl prüfen** und **Berechnung starten**.
5. Den gespeicherten Lauf öffnen, MP4 abspielen oder herunterladen und anhand
   eines Aufnahmezeitpunkts ein Einzelbild anzeigen und als PNG herunterladen.

Gespeicherte Datensatzregeln einschließlich ihres Samplings gelten zuerst.
Danach werden Dateien dedupliziert, nach Aufnahmezeitpunkt und Dateipfad sortiert,
auf den jeweiligen Zeitraum eingeschränkt und erneut gesampelt. Rate 15 wählt
Bild 15, 30, 45 …; ein unvollständiger Restblock entfällt. Zufall wählt exakt die
angegebene Anzahl ohne Zurücklegen. Derselbe Seed und dieselbe Ausgangsauswahl
liefern dieselben Dateien. Eine zu große Zufallsanzahl wird nicht still gekürzt.
Referenz- und Anomaliezeitraum dürfen überlappen.

Standards: Sampling 1, Seed 42, automatischer Kontrast und 10 FPS. Kontrast und
FPS gehören zur gespeicherten Konfiguration; für Änderungen einen Lauf **Als
Vorlage übernehmen** und neu berechnen.

## Bedeutung der Darstellung

Die Rechnung verwendet die Pipeline-Ausgabe ohne zusätzliche Normalisierung pro
Bild. Das Referenzbild ist deren arithmetischer Mittelwert in Gleitkommapräzision.
Die Bilddifferenz ist `aktuelles Bild − Referenzbild`:

- Keine Änderung: Grau (128).
- Positive Änderung: heller bis Weiß.
- Negative Änderung: dunkler bis Schwarz.

Die Darstellung beschreibt Helligkeitsänderungen, keine fachliche Interpretation.
Außerhalb des Einzelbilds steht die mittlere absolute Differenz als Bildabstand.

Automatischer Kontrast verwendet die größte absolute Pixeldifferenz aller
Anomalieframes als gemeinsame symmetrische Grenze. Manuell wird diese Grenze in
Einheiten der Pipeline-Ausgabe eingestellt; größere Abweichungen werden
abgeschnitten. Unveränderte Bilder bleiben auch bei automatischer Nullgrenze grau.

Jeder Frame enthält oben rechts den Aufnahmezeitpunkt. Das Video verwendet eine
konstante Bildrate ohne Auffüllen zeitlicher Lücken. Die Zeitpunkt-Suche liefert
den exakten Treffer oder den ersten späteren Frame; hinter dem letzten Zeitpunkt
den letzten Frame. Bei einer Abweichung zeigt die Oberfläche beide Zeitpunkte.

PNG und MP4 verwenden dieselben gerenderten Frames. PNG ist verlustfrei, MP4 wird
mit H.264/YUV420p und Faststart für Browser kodiert. Ungerade Bildabmessungen werden
am unteren/rechten Rand um höchstens einen Pixel ergänzt. Das Referenzbild steht
zusätzlich als PNG-Vorschau bereit; sein Gleitkommaoriginal bleibt intern erhalten.

## Speicherung und API

Läufe werden als CPU-Jobs im bestehenden Scheduler ausgeführt. Die Auswahl und die
Pipeline werden beim Start eingefroren; veränderte oder fehlende Quelldateien
führen zu einem Fehler. Bilder werden sequenziell verarbeitet; temporäre
Differenzdateien erlauben eine gemeinsame Skala ohne die Bildserie im RAM zu halten.

Die Migration `0061_reference_image` ergänzt `reference_image_runs`. Sie wird wie
die bestehenden Projektmigrationen beim Backendstart angewendet. Artefakte liegen
projektgebunden unter `reference_image_runs/<id>/`. Im Data Manager lassen sich
Läufe und ihre Abhängigkeiten prüfen und abgeschlossene Läufe löschen. Aktive
Läufe müssen zuerst abgebrochen werden. Nur fertige Ergebnisse sind abrufbar.

API-Basis: `/api/reference-image-analysis` mit bestehendem Projektkontext:

- `POST /preview`: verfügbare/ausgewählte Bilder und Validierungsfehler.
- `POST /runs`, `GET /runs`, `GET /runs/{id}`: Konfiguration und Laufstatus.
- `POST /runs/{id}/abort`, `DELETE /runs/{id}`, `GET /runs/{id}/log`.
- `GET /runs/{id}/results`: Zusammenfassung und Frameindex mit Zeitstempeln/Abständen.
- `GET /runs/{id}/frame?timestamp=...`: exakter oder nächster späterer Frame.
- `GET /runs/{id}/artifacts/{name}`: `video.mp4`, `reference.png`,
  `frame_000000.png` usw.; mit `download=true` als Download.

Medienlinks führen die Projekt-ID als `project_id` mit. Fehlgeschlagene und
abgebrochene Läufe veröffentlichen keine unvollständigen Artefakte.
