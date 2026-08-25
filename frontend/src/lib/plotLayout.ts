import type { Data, Layout } from './plotly';

function hasVisibleRangeSlider(layout: Partial<Layout>): boolean {
  const rangeSlider = (layout.xaxis as { rangeslider?: { visible?: boolean } } | undefined)?.rangeslider;
  return rangeSlider !== undefined && rangeSlider.visible !== false;
}

function hasVisibleLegend(data: Data[], layout: Partial<Layout>): boolean {
  if (layout.showlegend === false) return false;
  return data.some((input) => {
    const trace = input as unknown as { showlegend?: boolean; visible?: boolean | 'legendonly' };
    return trace.showlegend !== false && trace.visible !== false;
  });
}

/** Keep Plotly's overview range slider and bottom legend in separate bands. */
export function withSeparatedRangeSliderLegend(
  layout: Partial<Layout> | undefined,
  data: Data[],
): Partial<Layout> {
  const source = layout ?? {};
  if (!hasVisibleRangeSlider(source) || !hasVisibleLegend(data, source)) return source;
  const margin = source.margin ?? {};
  const legend = source.legend ?? {};
  return {
    ...source,
    margin: { ...margin, b: Math.max(Number(margin.b) || 0, 136) },
    legend: {
      ...legend,
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: -0.36,
      yanchor: 'top',
    },
  };
}
