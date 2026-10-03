import type { RoiRectangle } from './roiTypes';
export type Corner = 'tl' | 'tr' | 'bl' | 'br';
export type DragMode = Corner | 'move';
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, Math.round(value)));
export function dragRectangle(rect: RoiRectangle, mode: DragMode, dx: number, dy: number, width: number, height: number): RoiRectangle {
  if (mode === 'move') return {...rect, x: clamp(rect.x + dx, 0, width - rect.width), y: clamp(rect.y + dy, 0, height - rect.height)};
  let left = rect.x, right = rect.x + rect.width, top = rect.y, bottom = rect.y + rect.height;
  if (mode.endsWith('l')) left = clamp(left + dx, 0, right - 1);
  else right = clamp(right + dx, left + 1, width);
  if (mode.startsWith('t')) top = clamp(top + dy, 0, bottom - 1);
  else bottom = clamp(bottom + dy, top + 1, height);
  return {x: left, y: top, width: right - left, height: bottom - top};
}
export function roiCoordinates(rect: RoiRectangle) {
  return `Ecken (Pixelgrenzen): (${rect.x}, ${rect.y}), (${rect.x + rect.width}, ${rect.y}), (${rect.x + rect.width}, ${rect.y + rect.height}), (${rect.x}, ${rect.y + rect.height}) · Ausschnitt: ${rect.width} × ${rect.height} Pixel`;
}
