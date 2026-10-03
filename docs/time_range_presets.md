# Projektweite Zeitraumvorlagen

„Gespeicherte Zeiträume“ ist eine gemeinsame Bibliothek pro Projekt. Sie wird in
der Referenzbild-Analyse, im Varianzvergleich und in jeder Normal-
und Anomalie-/Ereigniszeile von Resolution Sensitivity verwendet.
Übernehmen kopiert ausschließlich Beginn und Ende. Sampling, Zufallsseed,
Datensatz und Preprocessing bleiben erhalten. Beide Grenzen zählen einschließlich;
Beginn gleich Ende ist erlaubt. Referenzbild-Analyse und Varianzvergleich
erlauben überlappende Phasen; Resolution Sensitivity verbietet Überschneidungen,
einschließlich eines gemeinsamen End-/Startzeitpunkts.

„Zeitraum speichern“ öffnet einen Dialog mit den aktuellen Zeiten und einem
Namensfeld. „Zeiträume bearbeiten / löschen“ bietet Suche, Details, Anlegen,
Umbenennen, Ändern der Zeiten und Löschen mit einer Bestätigung, die den Namen
der Vorlage nennt. Änderungen werden in allen eingebundenen
Auswahlfeldern desselben Projekts unmittelbar sichtbar. Im Data Manager steht
der Objekttyp „Gespeicherte Zeiträume“ zur Verfügung.

## Übersicht und Resolution Sensitivity

Im Bilddaten-Arbeitsbereich folgt der Reiter „Gespeicherte Zeiträume“ direkt auf
„Train/Test Datasets“. Die nach Namen sortierte, durchsuchbare Tabelle zeigt Name,
Beginn, Ende, Dauer sowie Erstellungs- und Änderungszeitpunkt. „Neuer Zeitraum“
benötigt keinen Datensatz; die Detailansicht zeigt zusätzlich die ID. Übersicht
und Verwaltungsdialog verwenden dieselben Formulare und Löschbestätigungen.

In Resolution Sensitivity legen die Hinzufügen-Schaltflächen weiterhin lokale
Intervallzeilen an. Eine Vorlage befüllt nur die Zeitwerte; Name, Rolle, Pipelines
und Sollzeitpunkte bleiben erhalten. „Aus Analyse entfernen“ entfernt ausschließlich
die Zeile. „Gespeicherten Zeitraum löschen“ entfernt die Bibliotheksvorlage.
Der Import aus Evaluation-Labelsets und die Auswahl über gleichmäßig verteilte
Sollzeitpunkte mit nächstgelegenen Bildern bleiben erhalten. Überschneidungen
werden vor dem Start an beiden betroffenen Zeilen angezeigt.

Neue Resolution-Sensitivity-Läufe speichern `interval_end_inclusive: true` im
bestehenden Konfigurations-JSON. Bei der Ausführung alter Läufe ohne Kennzeichen
(oder mit `false`) bleibt die Endgrenze ausschließlich. Exporte werden nicht neu
berechnet. Dafür sind keine zusätzlichen Datenbankfelder oder Migrationen nötig.

## Speicherung und API

Die Migration `0062_time_range_presets` folgt auf `0061_reference_image`.
`time_range_presets` liegt in der jeweiligen Projektdatenbank. Es gibt keine
Verknüpfung zu Datensätzen, Pipelines oder Analyse-Läufen. Der interne eindeutige
`name_key` ist der mit Python `casefold()` normalisierte Name; dadurch werden
auch Unicode-Groß-/Kleinschreibungsvarianten erkannt. Namen werden getrimmt,
sind 1–255 Zeichen lang und dürfen pro Projekt nur einmal vorkommen. Identische
Zeitwerte unter anderen Namen sind erlaubt.

Alle Anfragen verwenden `X-MLTrace-Project-ID` (die bestehende Middleware
unterstützt alternativ `project_id`).

| Methode | Pfad | Ergebnis |
| --- | --- | --- |
| GET | `/api/time-range-presets` | Liste, nach Namen sortiert |
| POST | `/api/time-range-presets` | Anlegen, 201 |
| PUT | `/api/time-range-presets/{id}` | Name und Zeiten vollständig ersetzen, 200 |
| DELETE | `/api/time-range-presets/{id}` | Löschen, 204 |

POST/PUT erwarten `{ "name": "Normalphase", "start": "2026-01-01T10:00:00", "end": "2026-01-01T11:00:00" }`.
Antwortobjekte enthalten `id`, `name`, `start`, `end`, `created_at`, `updated_at`.
Ungültige Eingaben liefern 422, Namenskonflikte 409 und fehlende IDs 404.

Beginn und Ende folgen der lokalen Bilddatensatz-Zeitkonvention. Es erfolgt
keine Zeitzonenumrechnung; ein mitgelieferter Offset wird wie bei den bestehenden
Datensatzformularen verworfen. Beispiel: `10:00:00+02:00` bleibt `10:00:00`.
Die Erstellungs-/Änderungsmetadaten verwenden die bestehende UTC-Konvention.

## Einbindung in weitere Reiter

Das eigenständige Modul `frontend/src/timeRangePresets` enthält Typen,
Validierung, einen gemeinsamen Store, den Hook `useTimeRangePresets` und die
Komponenten `TimeRangePresetPicker`, `TimeRangePresetManager` und
`TimeRangePresetActions`. HTTP-Funktionen stehen in `src/api.ts`.
Jeder HTTP-Aufruf erhält eine explizite Projekt-ID, damit ein noch laufender
Aufruf beim Projektwechsel immer an sein ursprüngliches Projekt gebunden bleibt.

```tsx
import { TimeRangePresetPicker } from '../timeRangePresets/TimeRangePresetPicker';
import { copyTimeRange } from '../timeRangePresets/helpers';

<TimeRangePresetPicker
  projectId={projectId}
  active={active}
  value={form.period}
  min={dataset?.start_timestamp ?? undefined}
  max={dataset?.end_timestamp ?? undefined}
  disabled={busy}
  applyDisabledReason={!dataset ? 'Bitte zuerst einen Datensatz auswählen.' : undefined}
  onApply={range => {
    setPreview(null);
    setForm(current => ({
      ...current,
      period: copyTimeRange(current.period, range),
    }));
  }}
/>
```

`value` und `onApply` verwenden `{ start: string; end: string }`. `min` und `max`
sind optionale einschließlich gültige Datensatzgrenzen. Unpassende Vorlagen
werden mit Erklärung deaktiviert und nie beschnitten. `applyDisabledReason`
sperrt nur das Übernehmen, etwa bis ein Datensatz gewählt wurde. `disabled`
sperrt alle Aktionen während einer Berechnung. `active` verhindert unnötiges
Laden bei ausgeblendeten Reitern und lädt beim erneuten Öffnen die aktuelle
Liste, auch nach Änderungen im Data Manager.

Der aufrufende Reiter invalidiert seine Auswahlvorschau sowohl bei `onApply`
(auch bei denselben Zeitwerten) als auch bei manuellen Änderungen. Ob im
Zeitraum tatsächlich Bilder vorhanden sind, prüft weiterhin die bestehende
Analyse-Auswahlprüfung. Die Komponente enthält keine Referenz-/Anomalielogik.

Alle Auswahlfelder verwenden denselben nach Projekt-ID getrennten Store.
Parallele Leseanfragen werden zusammengefasst. Eine veraltete Ladeantwort darf
keine zwischenzeitliche Änderung oder Löschung überschreiben. Ein Projektwechsel
setzt Dialoge zurück und zeigt ausschließlich Einträge des neuen Projekts.
Bearbeiten/Löschen ruft `onApply` niemals auf: Übernommene Formularwerte und
bereits gespeicherte Analyse-Konfigurationen bleiben unverändert.

Für eine eigenständige Übersicht kann `<TimeRangePresetManager projectId={projectId}
active={active} />` eingebunden werden. Der Manager enthält Tabelle, Suche und alle
Aktionen. `TimeRangePresetActions` rendert das gemeinsame Formular, die Detailansicht
oder die Löschbestätigung anhand von `action.mode` (`create`, `edit`, `details`,
`delete`); `onClose` und optional `onBusyChange` steuern den umgebenden Dialog.
Auch aufrufende Analyseseiten sollten mit `key={projectId}` gerendert werden, damit
beim Projektwechsel lokale Formulare und Dialoge zurückgesetzt werden.

## Prüfung

Backend: `backend/tests/test_time_range_presets.py` prüft CRUD, Migration und
Neuverbindung, Namensvalidierung inklusive Unicode, lokale Zeiten,
Projekttrennung, Data Manager und unveränderte Analyse-Läufe.
Frontend: `src/timeRangePresets/*.test.ts` prüft API-Routing, Zeitgrenzen,
Kopieren ohne andere Einstellungen, synchrone Listen, Ladefehler und
Projektwechsel einschließlich verspäteter Antworten.
`test_resolution_sensitivity.py` prüft zusätzlich inklusive Grenzen, Einzelzeitpunkte,
Überschneidungen und die Legacy-Auswahlregel. `resolutionSensitivity/helpers.test.ts`
prüft die betroffenen Zeilen, Datensatzgrenzen und unveränderte Einstellungen.
