import { useEffect, useRef } from 'react';
import { useMantineColorScheme } from '@mantine/core';
import Plotly, { type Data, type Layout, type Config, type PlotlyHTMLElement, type PlotMouseEvent, type PlotSelectionEvent } from '../lib/plotly';

export type PlotlyChartClick = {
  timestamp: string;
  curveNumber: number;
  pointNumber: number;
};

export type PlotlyChartSelection = {
  start: string;
  end: string;
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
export function PlotlyChart({ data, layout, config, height = 400, className, onClick, onSelected }: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const { colorScheme } = useMantineColorScheme();
  const dark = colorScheme === 'dark';

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;

    const themedLayout: Partial<Layout> = {
      autosize: true,
      margin: { l: 56, r: 24, t: 16, b: 48 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {
        family: 'Inter, system-ui, sans-serif',
        size: 12,
        color: dark ? '#c1c2c5' : '#343a40',
      },
      ...layout,
    };

    const plot = el as unknown as PlotlyHTMLElement;
    Plotly.react(plot, data, themedLayout, { ...BASE_CONFIG, ...config });

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

    const observer = new ResizeObserver(() => {
      Plotly.Plots.resize(el);
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
      plot.removeAllListeners('plotly_click');
      plot.removeAllListeners('plotly_selected');
      Plotly.purge(plot);
    };
  }, [data, layout, config, dark, onClick, onSelected]);

  return (
    <div
      ref={ref}
      className={className}
      style={{ width: '100%', height: typeof height === 'number' ? `${height}px` : height }}
    />
  );
}

export default PlotlyChart;
