import { describe, expect, it } from 'vitest';
import type { Data } from './plotly';
import { preparePlotData, withLineGapPolicy } from './plotGaps';

function line(x: unknown[], y: unknown[], extra: Record<string, unknown> = {}): Data {
  return { type: 'scatter', mode: 'lines', x, y, ...extra } as unknown as Data;
}

function prepared(trace: Data): Record<string, unknown> {
  return preparePlotData([trace])[0] as unknown as Record<string, unknown>;
}

describe('preparePlotData', () => {
  it('keeps explicit missing values empty and normalizes non-finite values', () => {
    const result = prepared(line([0, 1, 2, 3], [1, null, Number.NaN, Number.POSITIVE_INFINITY]));
    expect(result.y).toEqual([1, null, null, null]);
    expect(result.connectgaps).toBe(false);
    expect(result).not.toHaveProperty('marker');
    expect(result).not.toHaveProperty('error_x');
    expect(result).not.toHaveProperty('error_y');
    expect(result).not.toHaveProperty('customdata');
  });

  it('breaks when a continuity segment changes and keeps point metadata aligned', () => {
    const trace = withLineGapPolicy(
      line(
        ['2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z', '2026-01-01T00:00:02Z'],
        [1, 2, 3],
        { customdata: ['a', 'b', 'c'], text: ['A', 'B', 'C'] },
      ),
      { continuity: [0, 1, 1] },
    );
    const result = prepared(trace);
    expect(result.y).toEqual([1, null, 2, 3]);
    expect(result.customdata).toEqual(['a', null, 'b', 'c']);
    expect(result.text).toEqual(['A', null, 'B', 'C']);
    expect(result).not.toHaveProperty('mltraceGapPolicy');
  });

  it('keeps a single missing point and exactly fifteen seconds connected for fast cadence', () => {
    const result = prepared(line(
      [
        '2026-01-01T00:00:00.000Z',
        '2026-01-01T00:00:01.000Z',
        '2026-01-01T00:00:03.000Z',
        '2026-01-01T00:00:18.000Z',
        '2026-01-01T00:00:19.000Z',
        '2026-01-01T00:00:20.000Z',
      ],
      [0, 1, 2, 3, 4, 5],
    ));
    expect(result.y).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('breaks above fifteen seconds for fast cadence', () => {
    const result = prepared(line(
      [
        '2026-01-01T00:00:00Z',
        '2026-01-01T00:00:01Z',
        '2026-01-01T00:00:17Z',
        '2026-01-01T00:00:18Z',
        '2026-01-01T00:00:19Z',
      ],
      [0, 1, 2, 3, 4],
    ));
    expect(result.y).toEqual([0, 1, null, 2, 3, 4]);
  });

  it('uses the same strict five-step boundary for slower time cadence', () => {
    const exact = prepared(line(
      ['2026-01-01T00:00:00Z', '2026-01-01T00:00:10Z', '2026-01-01T00:01:00Z', '2026-01-01T00:01:10Z', '2026-01-01T00:01:20Z'],
      [0, 1, 2, 3, 4],
    ));
    const above = prepared(line(
      ['2026-01-01T00:00:00Z', '2026-01-01T00:00:10Z', '2026-01-01T00:01:01Z', '2026-01-01T00:01:11Z', '2026-01-01T00:01:21Z'],
      [0, 1, 2, 3, 4],
    ));
    expect(exact.y).toEqual([0, 1, 2, 3, 4]);
    expect(above.y).toEqual([0, 1, null, 2, 3, 4]);
  });

  it('breaks explicitly discrete numeric axes only above five expected steps', () => {
    const exact = prepared(withLineGapPolicy(line([0, 1, 6, 7], [10, 11, 16, 17]), { discreteStep: 1 }));
    const above = prepared(withLineGapPolicy(line([0, 1, 7, 8], [10, 11, 17, 18]), { discreteStep: 1 }));
    expect(exact.y).toEqual([10, 11, 16, 17]);
    expect(above.y).toEqual([10, 11, null, 17, 18]);
  });

  it('leaves marker-only traces unchanged', () => {
    const marker = { type: 'scatter', mode: 'markers', x: [0, 2], y: [1, 3] } as Data;
    expect(preparePlotData([marker])[0]).toEqual(marker);
  });
});
