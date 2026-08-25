import { describe, expect, it } from 'vitest';

import { correlationMatrixView, filteredPairRows, INCOMPLETE_CORRELATION_MESSAGE } from './helpers';

describe('redundancy result helpers', () => {
  const matrix = {
    variables: ['excluded', 'b', 'a'],
    leaf_order: ['a', 'b'],
    spearman: [[1, null, 0.2], [null, 1, -0.8], [0.2, -0.8, 1]],
    common_n: [[10, 2, 10], [2, 10, 9], [10, 9, 10]],
  };

  it('keeps N/A cells and exclusions in original order', () => {
    const view = correlationMatrixView(matrix, 'original', 'signed');
    expect(view.names).toEqual(['excluded', 'b', 'a']);
    expect(view.values[0][1]).toBeNull();
    expect(view.commonN[0][1]).toBe(2);
  });

  it('uses only cluster leaves and absolute values in clustered order', () => {
    const view = correlationMatrixView(matrix, 'clustered', 'absolute');
    expect(view.names).toEqual(['a', 'b']);
    expect(view.values).toEqual([[1, 0.8], [0.8, 1]]);
  });

  it('applies an inclusive freely chosen pair threshold', () => {
    expect(filteredPairRows([{ absolute_rho: 0.89 }, { absolute_rho: 0.9 }], 0.9)).toEqual([{ absolute_rho: 0.9 }]);
  });

  it('exposes the exact clustering exclusion explanation', () => {
    expect(INCOMPLETE_CORRELATION_MESSAGE).toBe('Excluded from clustering because the correlation matrix is incomplete.');
  });
});
