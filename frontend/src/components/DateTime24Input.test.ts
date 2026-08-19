import { describe, expect, it } from 'vitest';

import { normalizeTime } from './DateTime24Input';

describe('DateTime24Input', () => {
  it('normalizes valid 24-hour manual values to seconds', () => {
    expect(normalizeTime('00:00')).toBe('00:00:00');
    expect(normalizeTime('23:59:58')).toBe('23:59:58');
  });

  it('rejects AM/PM-shaped and out-of-range manual values', () => {
    expect(normalizeTime('11:00 PM')).toBeNull();
    expect(normalizeTime('24:00:00')).toBeNull();
    expect(normalizeTime('12:60:00')).toBeNull();
  });
});
