import { NumberInput } from '@mantine/core';
import type { ReactNode } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';

import type { NumericDraftState } from '../types';

type BufferedNumberInputProps = {
  value: unknown;
  label: ReactNode;
  min?: number;
  max?: number;
  integerOnly?: boolean;
  disabled?: boolean;
  description?: ReactNode;
  error?: ReactNode;
  onCommit: (value: number) => void;
  onDraftStateChange?: (state: NumericDraftState | null) => void;
};

type ParsedDraft = {
  valid: boolean;
  value?: number;
  message?: string;
};

function normalizeValue(value: unknown): string | number {
  return typeof value === 'number' || typeof value === 'string' ? value : '';
}

function sameDraft(left: unknown, right: unknown): boolean {
  return String(normalizeValue(left)) === String(normalizeValue(right));
}

function parseDraft(value: string | number, min?: number, max?: number, integerOnly?: boolean): ParsedDraft {
  if (value === '' || value === null || value === undefined) {
    return { valid: false, message: 'Enter a value.' };
  }

  const text = String(value).trim();
  if (text === '' || text === '-' || text === '+' || text === '.' || text.endsWith('.')) {
    return { valid: false, message: 'Enter a complete number.' };
  }

  const parsed = typeof value === 'number' ? value : Number(text);
  if (!Number.isFinite(parsed)) {
    return { valid: false, message: 'Enter a finite number.' };
  }
  if (integerOnly && !Number.isInteger(parsed)) {
    return { valid: false, message: 'Enter a whole number.' };
  }
  if (min !== undefined && parsed < min) {
    return { valid: false, message: `Value must be at least ${min}.` };
  }
  if (max !== undefined && parsed > max) {
    return { valid: false, message: `Value must be at most ${max}.` };
  }
  return { valid: true, value: parsed };
}

export function BufferedNumberInput({
  value,
  label,
  min,
  max,
  integerOnly,
  disabled,
  description,
  error,
  onCommit,
  onDraftStateChange,
}: BufferedNumberInputProps) {
  const committedValue = useMemo(() => normalizeValue(value), [value]);
  const [draft, setDraft] = useState<string | number>(committedValue);
  const [localError, setLocalError] = useState<string | null>(null);
  const draftStateCallback = useRef(onDraftStateChange);

  useEffect(() => {
    draftStateCallback.current = onDraftStateChange;
  }, [onDraftStateChange]);

  useEffect(() => {
    setDraft(committedValue);
    setLocalError(null);
    draftStateCallback.current?.(null);
  }, [committedValue]);

  useEffect(() => () => draftStateCallback.current?.(null), []);

  function updateDraft(next: string | number) {
    setDraft(next);
    const dirty = !sameDraft(next, committedValue);
    if (!dirty) {
      setLocalError(null);
      draftStateCallback.current?.(null);
      return;
    }
    const parsed = parseDraft(next, min, max, integerOnly);
    setLocalError(parsed.valid ? null : (parsed.message ?? 'Invalid number.'));
    draftStateCallback.current?.({
      dirty,
      valid: parsed.valid,
      message: parsed.message,
    });
  }

  function commitDraft() {
    const dirty = !sameDraft(draft, committedValue);
    if (!dirty) {
      setLocalError(null);
      draftStateCallback.current?.(null);
      return;
    }
    const parsed = parseDraft(draft, min, max, integerOnly);
    if (!parsed.valid || parsed.value === undefined) {
      setLocalError(parsed.message ?? 'Invalid number.');
      draftStateCallback.current?.({ dirty: true, valid: false, message: parsed.message });
      return;
    }
    setLocalError(null);
    draftStateCallback.current?.(null);
    onCommit(parsed.value);
  }

  function discardDraft() {
    setDraft(committedValue);
    setLocalError(null);
    draftStateCallback.current?.(null);
  }

  return (
    <NumberInput
      label={label}
      min={min}
      max={max}
      disabled={disabled}
      description={description}
      value={draft}
      error={error ?? localError ?? undefined}
      onChange={(next) => updateDraft(next)}
      onBlur={commitDraft}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          commitDraft();
          event.currentTarget.blur();
        }
        if (event.key === 'Escape') {
          event.preventDefault();
          discardDraft();
          event.currentTarget.blur();
        }
      }}
    />
  );
}
