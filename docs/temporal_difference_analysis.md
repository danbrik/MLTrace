# Zeitabstands-Analyse

Der Reiter **Zeitabstands-Analyse** vergleicht die zeitliche Bildänderung in einem
Referenzzeitraum und einem Vergleichszeitraum. Er verwendet einen Train/Test-Datensatz
und eine gespeicherte Preprocessing-Pipeline. Ein dort definierter Zuschnitt bestimmt
den Bildbereich. Es gibt keine zusätzliche ROI, Normalisierung oder Intensitätsskalierung.
Die Pipeline muss gleich große, endliche Graustufenbilder liefern.

## Auswahl und Berechnung

Beide Zeiträume sind frei wählbar und können aus den projektweiten Zeitraumvorlagen
übernommen werden. Beginn und Ende zählen einschließlich. Eine Referenzdauer von
24 Stunden wird nicht erzwungen. Das gespeicherte Dataset-Sampling wird zuerst auf
seinen ursprünglichen Regeln angewendet; erst danach wird auf die Zeiträume gefiltert.

Die Standardabstände sind **1, 2, 5, 15, 30, 60 Sekunden**. Positive ganze Sekunden
lassen sich hinzufügen und entfernen. Die Liste ist eindeutig und aufsteigend sortiert.

Für jedes Bild zum Zeitpunkt `t` wird für jeden Abstand `delta` ein Bild exakt bei
`t + delta` gesucht. Beide Bilder müssen im selben Zeitraum liegen. Es gibt keine
Toleranz, Rundung, Interpolation oder zusätzliche Stichprobe. Überlappende Paare
sind erlaubt. Die Anzahl fehlender Zielbilder enthält auch Ziele außerhalb des
gewählten Zeitraums. Ein Abstand ohne Paare bleibt in der Tabelle enthalten;
seine Statistik ist leer. Ein Zeitraum ohne ein einziges auswertbares Paar
verhindert den Start.

Identische Dateipfade werden dedupliziert. Verschiedene Dateien am gleichen
Zeitpunkt werden als mehrdeutig abgelehnt. Auswahlzeiten sind Ortszeiten in
`Europe/Berlin`; zum Bestimmen tatsächlicher Sekundenabstände werden sie nach UTC
umgerechnet. Mehrdeutige und nicht existente Ortszeiten werden mit dem betroffenen
Zeitpunkt gemeldet. In den Paarwerten bleiben Ortszeit und UTC-Zeit erhalten.

Die Messgröße eines Paares lautet:

```
d(t, delta) = mean(abs(float64(I(t + delta)) - float64(I(t))))
```

Sie wird über sämtliche Pixel der Pipeline-Ausgabe berechnet. Je Zeitraum und
Abstand werden Paaranzahl, Median, Q1, Q3 und `IQR = Q3 - Q1` gespeichert. Quantile
verwenden lineare Interpolation. Das Plot-Band zeigt **Q1 bis Q3**, nicht einen
symmetrischen Fehlerbalken um den Median.

## Gespeicherte Läufe

Beim Einreihen werden Dataset-Regeln, Preprocessing, Dateipfade, Dateigröße,
Änderungszeit und Paarzuordnung eingefroren. Ein CPU-Worker im vorhandenen Scheduler
berechnet alle Paare mit einem auf 128 MiB begrenzten Bildcache. Einzelne Bilder,
die den Cache überschreiten, werden ohne Cache verarbeitet. Bildstapel des gesamten
Zeitraums werden nicht im Arbeitsspeicher gehalten. Die exakten Quantile benötigen
nur die skalaren Paarwerte einer Zeitraum-/Abstandsgruppe.

Die additive Migration `0065_temporal_difference` ergänzt pro Projektdatenbank:

- `temporal_difference_runs`: Konfiguration, Snapshots, Status und Plot-Einstellungen;
- `temporal_difference_values`: Einzelwerte mit beiden Bildidentitäten und Zeitstempeln;
- `temporal_difference_summaries`: Statistik je Zeitraum und Abstand.

Einzelwerte werden in Blöcken gespeichert. Erst ein erfolgreich abgeschlossener
Lauf gibt Tabellen und Downloads frei. Fehlende oder veränderte Quellen, ungültige
Pipeline-Ausgaben und Abbrüche liefern kein vollständiges Ergebnis. Bei einem
Berechnungsfehler werden die bereits geschriebenen Werte entfernt.

Läufe unterstützen Fortschritt, Logs, Abbruch, Löschen und Übernahme als Vorlage.
Dataset-Abhängigkeiten werden in der Ressourcenverwaltung berücksichtigt.
Änderungen der Berechnungseinstellungen erfordern einen neuen Lauf und eine neue
Auswahlprüfung.

## Tabelle und Plot

Die Ergebnistabelle stellt beide Zeiträume je Abstand gegenüber. Einzelne Paarwerte
sind gefiltert nach Zeitraum und Abstand sowie mit Seitennavigation abrufbar. Beide
Tabellen stehen als CSV mit Kommatrenner, Dezimalpunkt und unquoted Zahlen bereit.
Leere Statistiken bleiben leere CSV-Zellen.

Der Plot verwendet eine lineare Sekundenachse, zwei Mediankurven und transparente
Q1–Q3-Bänder. Nicht auswertbare Abstände erzeugen Lücken, auch im Band. Titel,
Achsentitel, Achsengrenzen und Kurvenfarben sind nachträglich änderbar und werden
über **Darstellung speichern** pro Lauf gespeichert. Die Berechnungskonfiguration
und Messwerte werden dabei nicht geändert.

**Plot als PNG** und **Plot als SVG** exportieren die aktuelle Darstellung mit
weißem Hintergrund. PNG wird mit zweifacher Auflösung einer 1400 × 700 Darstellung
exportiert; SVG bleibt vektorbasiert. Gespeicherte Tabellen und Darstellungsänderungen
benötigen keinen Zugriff auf die Bildquellen.

## API

Basis: `/api/temporal-difference`; bestehender Projektkontext per
`X-MLTrace-Project-ID`, bei Download-Links per `project_id`.

- `POST /preview`: Auswahlprüfung für Dataset, Pipeline, beide Zeiträume und `deltas_seconds`.
- `POST /runs`, `GET /runs`, `GET /runs/{id}`: Erstellung und Laufverwaltung.
- `GET /runs/{id}/summary`: gespeicherte Statistik.
- `GET /runs/{id}/pairs?offset=0&limit=50&role=reference&delta=2`: paginierte Einzelwerte;
  beide Filter sind optional, höchstens 500 Zeilen je Anfrage.
- `PUT /runs/{id}/plot-settings`: Titel, Farben und automatische oder manuelle Achsengrenzen.
- `GET /runs/{id}/csv/summary`, `GET /runs/{id}/csv/pairs`: vollständige CSV-Tabellen.
- `GET /runs/{id}/log`, `POST /runs/{id}/abort`, `DELETE /runs/{id}`: Log und Lebenszyklus.

Ergebnisschnittstellen geben bei unvollständigen Läufen keine Teilergebnisse frei.
