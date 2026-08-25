import { describe, expect, it } from 'vitest';
import type { Data, Layout } from './plotly';
import { withSeparatedRangeSliderLegend } from './plotLayout';

const data = [{ type: 'scatter', mode: 'lines', name: 'Score', x: [0, 1], y: [1, 2] } as Data];

describe('withSeparatedRangeSliderLegend', () => {
  it('allocates distinct bottom bands for a range slider and legend', () => {
    const input: Partial<Layout> = {
      margin: { l: 40, b: 48 },
      legend: { orientation: 'h', y: -0.18 },
      xaxis: { rangeslider: { visible: true, thickness: 0.1 } },
    };
    const result = withSeparatedRangeSliderLegend(input, data);
    expect(result.margin).toMatchObject({ l: 40, b: 136 });
    expect(result.legend).toMatchObject({ orientation: 'h', y: -0.36, yanchor: 'top' });
    expect(input.margin?.b).toBe(48);
    expect(input.legend?.y).toBe(-0.18);
  });

  it('does not rewrite layouts without a range slider', () => {
    const input: Partial<Layout> = { legend: { orientation: 'h', y: -0.2 } };
    expect(withSeparatedRangeSliderLegend(input, data)).toBe(input);
  });

  it('does not reserve legend space when the legend is disabled', () => {
    const input: Partial<Layout> = {
      showlegend: false,
      margin: { b: 48 },
      xaxis: { rangeslider: { visible: true } },
    };
    expect(withSeparatedRangeSliderLegend(input, data)).toBe(input);
  });
});
