import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Button, Group, Modal, Select, Stack, Text, TextInput, useMantineColorScheme } from '@mantine/core';
import Plotly, { type Data, type Layout, type Config, type PlotlyHTMLElement, type PlotMouseEvent, type PlotSelectionEvent, type PlotRelayoutEvent } from '../lib/plotly';
import { buildPlotExportTable, normalizedDownloadName, plotTableToCsv, plotTableToParquet } from '../lib/plotExport';
import { preparePlotData } from '../lib/plotGaps';
import { withSeparatedRangeSliderLegend } from '../lib/plotLayout';
import { relayoutRange, visibleYRelayoutUpdate, type TimeSeriesAxisRange, type TimeSeriesTraceValues } from '../lib/timeSeriesViewport';

export type PlotlyChartClick = {
  timestamp: string;
  curveNumber: number;
  pointNumber: number;
};

export type PlotlyChartSelection = {
  start: string;
  end: string;
};

export type PlotlyChartDoubleClick = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type PlotlyChartProps = {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  /** CSS height of the plot container. */
  height?: number | string;
  className?: string;
  onClick?: (event: PlotlyChartClick) => void;
  onSelected?: (event: PlotlyChartSelection) => void;
  onRelayout?: (event: PlotRelayoutEvent) => void;
  /** Fit every Y axis to finite trace values inside the visible X range. */
  rescaleYOnVisibleX?: boolean;
  /** Return true to consume the double-click before Plotly applies its global reset. */
  onDoubleClick?: (event: PlotlyChartDoubleClick) => boolean;
};

const BASE_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: { format: 'png', scale: 2 },
};

/**
 * Duenner Wrapper um Plotly.react: responsiv via ResizeObserver, raeumt beim
 * Unmount mit Plotly.purge auf. Kein react-plotly.js (React-19-Kompatibilitaet).
 */
export function PlotlyChart({ data, layout, config, height = 400, className, onClick, onSelected, onRelayout, rescaleYOnVisibleX = false, onDoubleClick }: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const rescaleFrameRef = useRef<number | null>(null);
  const pendingXRangeRef = useRef<TimeSeriesAxisRange | null>(null);
  const applyingYRangeRef = useRef(0);
  const [exportOpened, setExportOpened] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'parquet'>('csv');
  const [exportName, setExportName] = useState('plot-data');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const { colorScheme } = useMantineColorScheme();
  const dark = colorScheme === 'dark';
  const preparedData = useMemo(() => preparePlotData(data), [data]);
  const exportTable = useMemo(() => buildPlotExportTable(data), [data]);
  const chromeLayout = useMemo(() => withSeparatedRangeSliderLegend(layout, preparedData), [layout, preparedData]);
  const defaultExportName = useMemo(() => {
    const title = layout?.title;
    const text = typeof title === 'string' ? title : title?.text;
    return typeof text === 'string' && text.trim() ? text.replace(/<[^>]*>/g, '').trim() : 'plot-data';
  }, [layout?.title]);
  const openExport = useCallback(() => {
    setExportName(defaultExportName);
    setExportError(null);
    setExportOpened(true);
  }, [defaultExportName]);
  const traceValues = useMemo<TimeSeriesTraceValues[]>(() => preparedData.flatMap((trace) => {
    const values = trace as unknown as {
      x?: Array<string | number | Date | null>;
      y?: Array<number | null>;
      yaxis?: string;
    };
    if (!Array.isArray(values.x) || !Array.isArray(values.y)) return [];
    return [{ x: values.x, y: values.y, yaxis: values.yaxis ?? 'y' }];
  }), [preparedData]);
  const effectiveConfig = useMemo<Partial<Config>>(() => {
    const downloadButton = {
      name: 'downloadPlotData',
      title: 'Download plot data',
      icon: Plotly.Icons.disk,
      click: openExport,
    };
    const configuredAdditions = (config?.modeBarButtonsToAdd ?? []).filter((button) => (
      typeof button === 'string' || button.name !== downloadButton.name
    ));
    const additions = rescaleYOnVisibleX
      ? [...configuredAdditions, 'autoScale2d' as const, downloadButton]
      : [...configuredAdditions, downloadButton];
    if (!rescaleYOnVisibleX) return { ...BASE_CONFIG, ...config, modeBarButtonsToAdd: additions };
    const removals = (config?.modeBarButtonsToRemove ?? BASE_CONFIG.modeBarButtonsToRemove ?? [])
      .filter((button) => button !== 'autoScale2d');
    return {
      ...BASE_CONFIG,
      ...config,
      scrollZoom: config?.scrollZoom ?? true,
      modeBarButtonsToAdd: additions,
      modeBarButtonsToRemove: removals,
    };
  }, [config, openExport, rescaleYOnVisibleX]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;

    const themedLayout: Partial<Layout> = {
      autosize: true,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      ...chromeLayout,
      margin: { l: 56, r: 24, t: 16, b: 48, ...chromeLayout.margin },
      font: {
        family: 'Inter, system-ui, sans-serif',
        size: 12,
        color: dark ? '#c1c2c5' : '#343a40',
        ...chromeLayout.font,
      },
    };

    Plotly.react(el as unknown as PlotlyHTMLElement, preparedData, themedLayout, effectiveConfig);
  }, [preparedData, chromeLayout, effectiveConfig, dark]);

  useEffect(() => {
    const plot = ref.current as unknown as PlotlyHTMLElement | null;
    if (!plot) return undefined;

    plot.removeAllListeners('plotly_click');
    plot.removeAllListeners('plotly_selected');
    plot.removeAllListeners('plotly_relayout');
    plot.removeAllListeners('plotly_relayouting');

    const applyVisibleYRange = (xRange: TimeSeriesAxisRange | null) => {
      const update = visibleYRelayoutUpdate(traceValues, xRange);
      if (Object.keys(update).length === 0) return;
      applyingYRangeRef.current += 1;
      void Plotly.relayout(plot, update).finally(() => {
        applyingYRangeRef.current = Math.max(0, applyingYRangeRef.current - 1);
      });
    };

    const scheduleVisibleYRange = (event: PlotRelayoutEvent) => {
      if (!rescaleYOnVisibleX) return;
      const xRange = relayoutRange(event, 'xaxis');
      if (xRange === undefined) return;
      pendingXRangeRef.current = xRange;
      if (rescaleFrameRef.current !== null) return;
      rescaleFrameRef.current = window.requestAnimationFrame(() => {
        rescaleFrameRef.current = null;
        applyVisibleYRange(pendingXRangeRef.current);
      });
    };
    if (onClick) {
      plot.on('plotly_click', (event: PlotMouseEvent) => {
        const point = event.points[0];
        if (!point || point.x === null || point.x === undefined) return;
        onClick({ timestamp: String(point.x), curveNumber: point.curveNumber, pointNumber: point.pointNumber });
      });
    }
    if (onSelected) {
      plot.on('plotly_selected', (event: PlotSelectionEvent) => {
        const xRange = event?.range?.x;
        if (!xRange || xRange.length < 2) return;
        onSelected({ start: String(xRange[0]), end: String(xRange[1]) });
      });
    }
    plot.on('plotly_relayouting', scheduleVisibleYRange);
    plot.on('plotly_relayout', (event: PlotRelayoutEvent) => {
      const xRange = rescaleYOnVisibleX ? relayoutRange(event, 'xaxis') : undefined;
      if (xRange !== undefined && rescaleFrameRef.current !== null) {
        window.cancelAnimationFrame(rescaleFrameRef.current);
        rescaleFrameRef.current = null;
      }
      if (xRange !== undefined) applyVisibleYRange(xRange);
      // Programmatic Y-only relayouts must not feed page-level handlers back
      // into the shared scaling path. A simultaneous user X event still wins.
      if (applyingYRangeRef.current === 0 || xRange !== undefined) onRelayout?.(event);
    });

    const handleDoubleClick = (event: MouseEvent) => {
      if (!onDoubleClick) return;
      const bounds = plot.getBoundingClientRect();
      const handled = onDoubleClick({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
        width: bounds.width,
        height: bounds.height,
      });
      if (handled) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    plot.addEventListener('dblclick', handleDoubleClick, true);

    return () => {
      plot.removeAllListeners('plotly_click');
      plot.removeAllListeners('plotly_selected');
      plot.removeAllListeners('plotly_relayout');
      plot.removeAllListeners('plotly_relayouting');
      if (rescaleFrameRef.current !== null) {
        window.cancelAnimationFrame(rescaleFrameRef.current);
        rescaleFrameRef.current = null;
      }
      plot.removeEventListener('dblclick', handleDoubleClick, true);
    };
  }, [onClick, onDoubleClick, onRelayout, onSelected, rescaleYOnVisibleX, traceValues]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const observer = new ResizeObserver(() => Plotly.Plots.resize(el));
    observer.observe(el);

    return () => {
      observer.disconnect();
      Plotly.purge(el);
    };
  }, []);

  const download = async () => {
    if (exportTable.columns.length === 0) return;
    setExporting(true);
    setExportError(null);
    try {
      const filename = normalizedDownloadName(exportName, exportFormat);
      const blob = exportFormat === 'csv'
        ? new Blob([plotTableToCsv(exportTable)], { type: 'text/csv;charset=utf-8' })
        : new Blob([await plotTableToParquet(exportTable)], { type: 'application/vnd.apache.parquet' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setExportOpened(false);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'The plot data could not be exported.');
    } finally {
      setExporting(false);
    }
  };

  return (
    <>
      <div
        ref={ref}
        className={className}
        style={{ width: '100%', height: typeof height === 'number' ? `${height}px` : height }}
      />
      <Modal opened={exportOpened} onClose={() => !exporting && setExportOpened(false)} title="Download plot data" centered>
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            Every graph is exported as a separate column. Time-series rows use dataset-local timestamps in YYYY.MM.DD hh:mm:ss format.
          </Text>
          {exportTable.columns.length === 0 && (
            <Alert color="yellow" title="No tabular plot data">
              This visualization has no exportable X/Y or heatmap values.
            </Alert>
          )}
          {exportError && <Alert color="red" title="Export failed">{exportError}</Alert>}
          <TextInput label="File name" value={exportName} onChange={(event) => setExportName(event.currentTarget.value)} />
          <Select
            label="File type"
            data={[{ value: 'csv', label: 'CSV' }, { value: 'parquet', label: 'Parquet' }]}
            value={exportFormat}
            allowDeselect={false}
            onChange={(value) => setExportFormat(value === 'parquet' ? 'parquet' : 'csv')}
          />
          <Text size="xs" c="dimmed">
            {exportTable.rowCount.toLocaleString()} rows · {exportTable.seriesCount.toLocaleString()} graph columns
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setExportOpened(false)} disabled={exporting}>Cancel</Button>
            <Button onClick={() => void download()} loading={exporting} disabled={exportTable.columns.length === 0}>Download</Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}

export default PlotlyChart;
