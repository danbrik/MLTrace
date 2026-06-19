import { useEffect, useRef } from 'react';
import { useMantineColorScheme } from '@mantine/core';
import Plotly, { type Data, type Layout, type Config } from '../lib/plotly';

type PlotlyChartProps = {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  /** CSS height of the plot container. */
  height?: number | string;
  className?: string;
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
export function PlotlyChart({ data, layout, config, height = 400, className }: PlotlyChartProps) {
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

    Plotly.react(el, data, themedLayout, { ...BASE_CONFIG, ...config });

    const observer = new ResizeObserver(() => {
      Plotly.Plots.resize(el);
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
      Plotly.purge(el);
    };
  }, [data, layout, config, dark]);

  return (
    <div
      ref={ref}
      className={className}
      style={{ width: '100%', height: typeof height === 'number' ? `${height}px` : height }}
    />
  );
}

export default PlotlyChart;
