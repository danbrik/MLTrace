import { Alert, Badge, Button, Group, Loader, SegmentedControl, SimpleGrid, Stack, Text } from '@mantine/core';
import { Check, MousePointer2, Plus } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { Shape } from 'plotly.js';

import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import type { EvaluationLabelEvent, EvaluationScorePreview, EvaluationTimeRange } from '../types';
import { applyPendingTimelineSelection, eventShapes, resolveTimelineSelection, scoreTrace, type TimelineSelectionMode } from './helpers';

export function EvaluationTimeline({
  preview,
  loading,
  error,
  range,
  onRangeChange,
  events = [],
  onCreateEvent,
  disabled = false,
  title = 'Score timeline',
  selectionActionLabel = 'Use selection as range',
}: {
  preview: EvaluationScorePreview | null;
  loading: boolean;
  error?: string | null;
  range: EvaluationTimeRange;
  onRangeChange: (range: EvaluationTimeRange) => void;
  events?: EvaluationLabelEvent[];
  onCreateEvent?: (type: 'target' | 'exclusion', range: EvaluationTimeRange) => void;
  disabled?: boolean;
  title?: string;
  selectionActionLabel?: string;
}) {
  const [selectionMode, setSelectionMode] = useState<TimelineSelectionMode>('range');
  const [pendingSelection, setPendingSelection] = useState<EvaluationTimeRange | null>(null);

  const shapes = useMemo(() => {
    const result = eventShapes(events);
    if (pendingSelection) {
      result.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: pendingSelection.start_timestamp,
        x1: pendingSelection.end_timestamp,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(34,139,230,0.12)',
        line: { color: '#228be6', width: 1 },
        layer: 'below',
      } as Shape);
    }
    return result;
  }, [events, pendingSelection]);

  function handleSelection(selection: PlotlyChartSelection) {
    const decision = resolveTimelineSelection(selectionMode, selection.start, selection.end);
    if (decision.pendingRange) setPendingSelection(decision.pendingRange);
    if (decision.annotation) onCreateEvent?.(decision.annotation.type, decision.annotation);
  }

  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <DateTime24Input
          label="Start"
          value={range.start_timestamp}
          max={(range.end_timestamp || preview?.end_timestamp) ?? undefined}
          disabled={disabled}
          onChange={(value) => onRangeChange({ ...range, start_timestamp: value })}
        />
        <DateTime24Input
          label="End"
          value={range.end_timestamp}
          min={(range.start_timestamp || preview?.start_timestamp) ?? undefined}
          disabled={disabled}
          onChange={(value) => onRangeChange({ ...range, end_timestamp: value })}
        />
      </SimpleGrid>
      <Group justify="space-between" align="center">
        <div>
          <Text fw={600} size="sm">{title}</Text>
          <Text size="xs" c="dimmed">Zoom and pan only change the view. Drag horizontally to make a selection.</Text>
        </div>
        <Group gap="xs">
          {preview?.decimated && <Badge color="blue" variant="light">Preview decimated to {preview.points.length.toLocaleString()} / {preview.total.toLocaleString()} points</Badge>}
          <SegmentedControl
            size="xs"
            value={selectionMode}
            onChange={(value) => setSelectionMode(value as TimelineSelectionMode)}
            data={[
              { value: 'range', label: 'Range' },
              { value: 'target', label: 'Target event' },
              { value: 'exclusion', label: 'Exclusion' },
            ]}
            disabled={disabled || !onCreateEvent}
          />
        </Group>
      </Group>
      {loading && <Group justify="center" py="xl"><Loader size="sm" /><Text size="sm">Loading score preview…</Text></Group>}
      {error && <Alert color="red">{error}</Alert>}
      {!loading && !error && preview && preview.points.length > 0 && (
        <PlotlyChart
          data={[scoreTrace(preview.points)]}
          layout={{
            dragmode: 'select',
            hovermode: 'x unified',
            shapes,
            showlegend: false,
            uirevision: `${preview.testing_run_id}:${preview.score_series}`,
            xaxis: {
              type: 'date',
              rangeslider: { visible: true, thickness: 0.1 },
              title: { text: 'Dataset-local time' },
            },
            yaxis: { title: { text: preview.score_series } },
          }}
          config={{ scrollZoom: true, modeBarButtonsToAdd: ['select2d'] }}
          height={430}
          onSelected={handleSelection}
        />
      )}
      {!loading && !error && (!preview || preview.points.length === 0) && (
        <Alert color="gray" icon={<MousePointer2 size={16} />}>Select a source to load its score preview.</Alert>
      )}
      {pendingSelection && selectionMode === 'range' && (
        <Group justify="space-between" p="xs" style={{ border: '1px solid var(--mantine-color-blue-4)', borderRadius: 4 }}>
          <Text size="sm">Selected: {pendingSelection.start_timestamp.replace('T', ' ')} – {pendingSelection.end_timestamp.replace('T', ' ')}</Text>
          <Group gap="xs">
            <Button size="compact-sm" variant="subtle" color="gray" onClick={() => setPendingSelection(null)}>Clear</Button>
            <Button
              size="compact-sm"
              leftSection={<Check size={14} />}
              onClick={() => {
                onRangeChange(applyPendingTimelineSelection(range, pendingSelection));
                setPendingSelection(null);
              }}
              disabled={disabled}
            >
              {selectionActionLabel}
            </Button>
          </Group>
        </Group>
      )}
      {selectionMode !== 'range' && onCreateEvent && (
        <Text size="xs" c="dimmed"><Plus size={12} style={{ verticalAlign: 'middle' }} /> The next horizontal selection creates a {selectionMode === 'target' ? 'target event' : 'dataset exclusion'}.</Text>
      )}
    </Stack>
  );
}
