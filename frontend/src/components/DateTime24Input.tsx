import { Input, SimpleGrid, TextInput } from '@mantine/core';
import { useEffect, useMemo, useState } from 'react';
import type React from 'react';

type DateTime24InputProps = {
  label?: string;
  value: string;
  min?: string;
  max?: string;
  disabled?: boolean;
  error?: React.ReactNode;
  description?: React.ReactNode;
  onChange: (value: string) => void;
};

function splitDateTime(value: string): { date: string; time: string } {
  const [date = '', time = ''] = value.slice(0, 19).split('T');
  return { date, time };
}

function normalizeTime(value: string): string | null {
  const match = value.trim().match(/^(\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  const seconds = Number(match[3] ?? '0');
  if (hours > 23 || minutes > 59 || seconds > 59) return null;
  return `${match[1]}:${match[2]}:${String(seconds).padStart(2, '0')}`;
}

// Native datetime-local controls can render AM/PM depending on browser locale.
// This component keeps the calendar native but makes the time field explicit 00-23.
export function DateTime24Input({
  label,
  value,
  min,
  max,
  disabled,
  error,
  description,
  onChange,
}: DateTime24InputProps) {
  const current = useMemo(() => splitDateTime(value), [value]);
  const minParts = useMemo(() => splitDateTime(min ?? ''), [min]);
  const maxParts = useMemo(() => splitDateTime(max ?? ''), [max]);
  const [timeDraft, setTimeDraft] = useState(current.time);

  useEffect(() => {
    setTimeDraft(current.time);
  }, [current.time]);

  const timeError = timeDraft && !normalizeTime(timeDraft) ? 'Use 00-23 time, e.g. 16:30:00' : undefined;

  return (
    <div>
      {label && (
        <Input.Label mb={4}>
          {label}
        </Input.Label>
      )}
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="xs">
        <TextInput
          aria-label={label ? `${label} date` : 'Date'}
          type="date"
          value={current.date}
          min={minParts.date}
          max={maxParts.date}
          disabled={disabled}
          error={error}
          onChange={(event) => {
            const nextDate = event.currentTarget.value;
            if (!nextDate) {
              onChange('');
              return;
            }
            onChange(`${nextDate}T${normalizeTime(timeDraft) ?? current.time ?? '00:00:00'}`);
          }}
        />
        <TextInput
          aria-label={label ? `${label} time` : 'Time'}
          value={timeDraft}
          placeholder="HH:mm:ss"
          disabled={disabled}
          error={timeError}
          onChange={(event) => {
            const nextTime = event.currentTarget.value;
            setTimeDraft(nextTime);
            const normalized = normalizeTime(nextTime);
            if (current.date && normalized) onChange(`${current.date}T${normalized}`);
          }}
          onBlur={() => {
            if (timeDraft && !normalizeTime(timeDraft)) setTimeDraft(current.time);
          }}
        />
      </SimpleGrid>
      {description && (
        <Input.Description mt={4}>
          {description}
        </Input.Description>
      )}
    </div>
  );
}
