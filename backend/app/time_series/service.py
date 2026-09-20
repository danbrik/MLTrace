import csv
import io
import re

import pandas as pd
from sqlalchemy.orm import Session

from app.models import TimeSeriesDataset, TimeSeriesSplit
from app.time_series.schemas import DatasetImport, DatasetUpdate, SplitInput

MAX_CSV_BYTES = 50 * 1024 * 1024


def read_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("Die CSV darf höchstens 50 MB groß sein.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Bitte die CSV mit UTF-8-Kodierung speichern.") from exc
    if not text.strip() or "\x00" in text:
        raise ValueError("Die Datei ist leer oder keine gültige CSV.")
    try:
        first_line = text.splitlines()[0]
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t|")
        reader = csv.reader(io.StringIO(text, newline=""), dialect, strict=True)
        headers = next(reader)
        if len(headers) < 2 or any(not h.strip() for h in headers) or len(headers) != len(set(headers)):
            raise ValueError("Die CSV benötigt mindestens zwei eindeutige, nicht leere Spaltennamen.")
        rows = []
        for row in reader:
            if not row:  # CSV blank lines are not records.
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV-Zeile {reader.line_num}: Anzahl der Werte passt nicht zu den Spalten.")
            rows.append(row)
    except (csv.Error, StopIteration) as exc:
        raise ValueError("CSV nicht lesbar. Unterstützte Trennzeichen: Komma, Semikolon, Tab und |.") from exc
    if not rows:
        raise ValueError("Die CSV enthält keine Datenzeilen.")
    return pd.DataFrame(rows, columns=headers)


def parse_times(frame: pd.DataFrame, column: str, fmt: str) -> pd.DatetimeIndex:
    if column not in frame.columns:
        raise ValueError("Bitte eine vorhandene Zeitspalte auswählen.")
    values = frame[column]
    try:
        if fmt in {"unix_s", "unix_ms", "unix_us", "unix_ns"}:
            values = pd.to_numeric(values, errors="raise")
            parsed = pd.to_datetime(values, unit=fmt[5:], utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(values, format=fmt or "ISO8601", utc=True, errors="coerce")
        times = pd.DatetimeIndex(parsed).as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Zeitspalte und Zeitformat passen nicht zusammen.") from exc
    invalid = times.isna()
    if invalid.any():
        examples = ", ".join(str(i + 2) for i, bad in enumerate(invalid) if bad)[:100]
        raise ValueError(f"{invalid.sum()} ungültige oder leere Zeitstempel (CSV-Zeilen: {examples}). Bitte Zeitformat prüfen.")
    return times


def iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def label_split(frame: pd.DataFrame, times: pd.DatetimeIndex, label_column: str) -> dict:
    if label_column not in frame.columns:
        raise ValueError("Bitte eine vorhandene Labelspalte auswählen.")
    labels = frame[label_column].str.strip().str.casefold()
    allowed = {"normal", "before_anomaly", "anomaly", "cooldown"}
    invalid = ~labels.isin(allowed)
    if invalid.any():
        examples = ", ".join(f"{i + 2}: {frame[label_column].iloc[i]!r}" for i in range(len(frame)) if invalid.iloc[i])[:300]
        raise ValueError(f"Ungültige oder leere Labels (CSV-Zeilen {examples}). Erlaubt: normal, before_anomaly, anomaly, cooldown.")
    duplicates = times.duplicated(keep=False)
    if duplicates.any():
        examples = ", ".join(f"{i + 2}: {iso(times[i])}" for i in range(len(times)) if duplicates[i])[:300]
        raise ValueError(f"Doppelte Zeitstempel verhindern den automatischen Split (CSV-Zeilen {examples}).")
    order = times.argsort()
    intervals = []
    previous = None
    for index in order:
        label, timestamp = labels.iloc[index], iso(times[index])
        if label != previous:
            intervals.append({"id": f"label-{len(intervals) + 1}", "start": timestamp, "end": timestamp,
                              "subset": "train" if label == "normal" else "test",
                              "tags": [] if label == "normal" else [label], "row_count": 0})
        intervals[-1]["end"] = timestamp
        intervals[-1]["row_count"] += 1
        previous = label
    return {"tags": [label for label in ("before_anomaly", "anomaly", "cooldown") if label in set(labels)],
            "intervals": intervals,
            "counts": {subset: {"rows": sum(i["row_count"] for i in intervals if i["subset"] == subset),
                                "intervals": sum(i["subset"] == subset for i in intervals)}
                       for subset in ("train", "test", "validation")}}


def preview(content: bytes, column: str | None = None, fmt: str = "ISO8601", label_column: str | None = None) -> dict:
    frame = read_csv(content)
    result = {"columns": list(frame.columns), "row_count": len(frame), "rows": frame.head(12).values.tolist()}
    candidates = [c for c in frame.columns if c.strip().casefold() == "label"]
    result["detected_label_column"] = candidates[0] if len(candidates) == 1 else None
    if column:
        times = parse_times(frame, column, fmt)
        result.update(start=iso(times.min()), end=iso(times.max()))
        if label_column:
            if label_column == column:
                raise ValueError("Zeitspalte und Labelspalte müssen verschieden sein.")
            result["label_split"] = label_split(frame, times, label_column)
    elif label_column:
        raise ValueError("Für die Split-Vorschau zuerst die Zeitspalte auswählen.")
    return result


def validate_columns(columns: list[str], selected: list[str], timestamp: str, label_column: str | None = None):
    if len(selected) != len(set(selected)) or not set(selected) <= set(columns):
        raise ValueError("Die Spaltenauswahl enthält unbekannte oder doppelte Spalten.")
    if timestamp not in selected or len(selected) < 2:
        raise ValueError("Die Zeitspalte und mindestens eine Datenspalte müssen ausgewählt bleiben.")
    if label_column is not None:
        if label_column not in columns or label_column == timestamp:
            raise ValueError("Die Labelspalte muss vorhanden und von der Zeitspalte verschieden sein.")
        if label_column in selected:
            raise ValueError("Die Labelspalte ist eine Annotation und darf nicht als Sensorspalte ausgewählt werden.")


def create_dataset(db: Session, content: bytes, filename: str, payload: DatasetImport):
    frame = read_csv(content)
    validate_columns(list(frame.columns), payload.selected_columns, payload.timestamp_column, payload.label_column)
    times = parse_times(frame, payload.timestamp_column, payload.timestamp_format)
    if payload.auto_split and not payload.label_column:
        raise ValueError("Für den automatischen Split ist eine Labelspalte erforderlich.")
    generated = label_split(frame, times, payload.label_column) if payload.auto_split else None
    dataset = TimeSeriesDataset(
        name=payload.name, filename=filename, source_csv=content,
        columns=list(frame.columns), selected_columns=payload.selected_columns,
        timestamp_column=payload.timestamp_column, timestamp_format=payload.timestamp_format,
        label_column=payload.label_column,
        row_count=len(frame), start=iso(times.min()), end=iso(times.max()),
    )
    try:
        db.add(dataset)
        db.flush()
        if generated:
            split_name = payload.split_name or f"{payload.name[:241]} – Label-Split"
            save_split(db, dataset, SplitInput(name=split_name, dataset_id=dataset.id,
                       tags=generated["tags"], intervals=generated["intervals"]), commit=False)
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(dataset)
    return dataset


def update_dataset(db: Session, dataset: TimeSeriesDataset, payload: DatasetUpdate):
    validate_columns(dataset.columns, payload.selected_columns, dataset.timestamp_column, dataset.label_column)
    dataset.name = payload.name
    dataset.selected_columns = payload.selected_columns
    db.commit()
    db.refresh(dataset)
    return dataset


def dataset_read(dataset: TimeSeriesDataset, *, detail=False):
    result = {key: getattr(dataset, key) for key in (
        "id", "name", "filename", "columns", "selected_columns", "timestamp_column", "timestamp_format",
        "row_count", "start", "end", "created_at", "updated_at", "label_column",
    )}
    result["split_count"] = len(dataset.splits)
    if detail:
        frame = read_csv(dataset.source_csv)
        # Active preview plus original sample for reversible column selection.
        result["rows"] = frame[dataset.selected_columns].head(12).values.tolist()
        result["source_rows"] = frame.head(12).values.tolist()
    return result


def boundary(value: str) -> pd.Timestamp:
    if not re.match(r"^\d{4}-\d{2}-\d{2}(?:T| )\d{2}:\d{2}", value):
        raise ValueError("Zeitraumgrenzen müssen Datum und Uhrzeit im ISO-Format enthalten.")
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError()
        result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
        return result.as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Ungültige Zeitraumgrenze.") from exc


def save_split(db: Session, dataset: TimeSeriesDataset, payload: SplitInput, split: TimeSeriesSplit | None = None, *, commit=True):
    if split and split.dataset_id != payload.dataset_id:
        raise ValueError("Die Datenbasis eines gespeicherten Splits kann nicht gewechselt werden.")
    frame = read_csv(dataset.source_csv)
    times = parse_times(frame, dataset.timestamp_column, dataset.timestamp_format).sort_values()
    intervals = []
    seen = set()
    for entry in payload.intervals:
        if entry.id in seen:
            raise ValueError("Zeiträume benötigen eindeutige IDs.")
        seen.add(entry.id)
        if not set(entry.tags) <= set(payload.tags) or len(entry.tags) != len(set(entry.tags)):
            raise ValueError("Ein Zeitraum enthält unbekannte oder doppelte Tags.")
        start, end = boundary(entry.start), boundary(entry.end)
        if start > end:
            raise ValueError("Der Beginn darf nicht nach dem Ende liegen.")
        if start < times[0] or end > times[-1]:
            raise ValueError("Zeiträume müssen innerhalb der Datenbasis liegen.")
        count = int(times.searchsorted(end, side="right") - times.searchsorted(start, side="left"))
        if not count:
            raise ValueError(f"Der Zeitraum {iso(start)} bis {iso(end)} enthält keine Datenzeile.")
        intervals.append({**entry.model_dump(), "start": iso(start), "end": iso(end), "row_count": count})
    intervals.sort(key=lambda item: boundary(item["start"]))
    for previous, current in zip(intervals, intervals[1:]):
        if boundary(current["start"]) <= boundary(previous["end"]):
            raise ValueError("Zeiträume dürfen sich nicht überschneiden – auch nicht zwischen Train, Test und Validation. Start und Ende sind inklusive.")
    if split is None:
        split = TimeSeriesSplit(dataset_id=dataset.id)
        db.add(split)
    split.name, split.tags, split.intervals = payload.name, payload.tags, intervals
    if commit:
        db.commit()
        db.refresh(split)
    else:
        db.flush()
    return split


def split_read(split: TimeSeriesSplit):
    return {**{key: getattr(split, key) for key in (
        "id", "dataset_id", "name", "tags", "intervals", "created_at", "updated_at",
    )}, "dataset_name": split.dataset.name}
