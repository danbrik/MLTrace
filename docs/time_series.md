# Zeitreihen-Datenbasis und Splits

Im Projektkopf lässt sich zwischen **Bilddaten** und **Zeitreihen** wechseln. Der Zeitreihenbereich enthält die Menüpunkte **Datenbasis** und **Splits**. Beide sind projektbezogen; CSV-Inhalte und Konfigurationen werden dauerhaft in der Projektdatenbank gespeichert.

## CSV importieren

1. Unter **Datenbasis → CSV importieren** eine UTF-8-CSV auswählen (maximal 50 MB; Komma, Semikolon, Tab oder Pipe als Trennzeichen).
2. Einen Namen vergeben und die Zeitspalte sowie das passende Zeitformat auswählen. ISO-Zeitstempel, deutsche Datumsformate, Unix-Zeit (s/ms/us/ns) und benutzerdefinierte `strptime`-Formate werden unterstützt.
3. **Zeitspalte prüfen** kontrolliert alle Zeilen und zeigt den verfügbaren Zeitraum. Leere oder ungültige Zeitstempel verhindern den Import. Zeitstempel ohne Zeitzone werden als UTC interpretiert; andere Zeitzonen werden nach UTC umgerechnet. Nanosekunden bleiben erhalten.
4. Nicht benötigte Spalten abwählen. Die Zeitspalte und mindestens eine weitere Spalte müssen aktiv bleiben.
5. **Datenbasis speichern**.

Gespeicherte Datenbasen lassen sich anzeigen, umbenennen, in ihrer Spaltenauswahl bearbeiten und löschen. Die Original-CSV bleibt gespeichert, damit abgewählte Spalten später wieder aktiviert werden können. Die Vorschau zeigt ausschließlich aktive Spalten. Spaltenänderungen gelten auch für bestehende Splits. Zeitspalte und Zeitformat bleiben nach dem Import fest, damit bestehende Zeiträume gültig bleiben. Für eine andere Zeitinterpretation eine neue Datenbasis importieren.

## Splits definieren

1. Unter **Splits → Neuer Split** eine Datenbasis wählen und den Split benennen.
2. Beliebige Tags anlegen, z. B. **Anomalie**, **Puffer** und **Normalzustand**. Jeder Zeitraum kann keine, eine oder mehrere dieser Tags tragen. Wird ein Tag entfernt, wird es auch aus allen Zeiträumen des bearbeiteten Splits entfernt.
3. Beginn und Ende in UTC im Format `JJJJ-MM-TTTHH:MM:SS` eingeben (optionale Sekundenbruchteile), **Train**, **Test** oder **Validation** und die Tags wählen, dann den Zeitraum hinzufügen. Die ersten/letzten Zeitstempel lassen sich per Schaltfläche übernehmen.
4. Weitere Zeiträume hinzufügen und **Split speichern**.

**Beide Grenzen sind inklusive.** Ein gemeinsamer Grenzzeitpunkt zählt bereits als Überschneidung. Innerhalb eines Splits darf kein Zeitraum mehrfach zugeordnet sein, auch nicht innerhalb derselben Gruppe oder bei verschiedenen Tags. Die Oberfläche sperrt Überschneidungen sofort; der Server prüft sie vor jedem Speichern erneut. Jeder Zeitraum muss innerhalb der Datenbasis liegen und mindestens eine Datenzeile enthalten. Mehrere Zeilen mit identischem Zeitstempel werden gemeinsam zugeordnet. Nicht zugeordnete Zeilen bleiben ungenutzt.

Mehrere benannte Splits können dieselbe Datenbasis für alternative Aufteilungen verwenden. Die Überschneidungsprüfung gilt jeweils innerhalb eines Splits. Die Datenbasis eines gespeicherten Splits bleibt fest.

Gespeicherte Splits werden in separaten Boxen für **Train**, **Test** und **Validation** mit chronologisch sortierten Zeiträumen, Tags und Zeilenzahlen angezeigt. Sie können erneut geöffnet, bearbeitet und gelöscht werden. Das Löschen einer Datenbasis entfernt nach Bestätigung auch ihre Splits.

## Implementierung und Prüfungen

- Migration `0058_time_series` ergänzt `time_series_datasets` und `time_series_splits`. Die normale Projektmigration beim Backend-Start übernimmt das Update.
- API: `/api/time-series/preview`, `/api/time-series/datasets` und `/api/time-series/splits`; Detail-, Bearbeitungs- und Löschoperationen über `/{id}`.
- Die Projektzuordnung nutzt den vorhandenen Header `X-MLTrace-Project-ID`.
- Transaktionale Speicherung hält Originaldatei, Metadaten und Änderungen konsistent. Listen laden die CSV-Blobs nicht mit.
- Tests: `pytest backend/tests/test_time_series.py` und `npm test -- src/timeSeries/intervals.test.ts` (im Frontend-Verzeichnis).
