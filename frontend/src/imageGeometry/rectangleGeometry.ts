export type LegacyRoiRectangle = { x: number; y: number; width: number; height: number };
export type RotatedRoiRectangle = { version: 2; center_x: number; center_y: number; width: number; height: number; angle_degrees: number };
export type RoiRectangle = LegacyRoiRectangle | RotatedRoiRectangle;
export type Corner = 'tl' | 'tr' | 'bl' | 'br';
export type DragMode = Corner | 'move' | 'rotate';
export type Point = {x: number; y: number};
const epsilon = 1e-9;
export function orientedRoi(rect: RoiRectangle): RotatedRoiRectangle {
  return 'version' in rect ? {...rect} : {version: 2, center_x: rect.x + rect.width / 2, center_y: rect.y + rect.height / 2, width: rect.width, height: rect.height, angle_degrees: 0};
}
export const normalizeAngle = (angle: number) => ((angle + 180) % 360 + 360) % 360 - 180;
function rotation(angle: number) {
  const snap = (value: number) => Math.abs(value - Math.round(value)) < 1e-14 ? Math.round(value) : value;
  const radians = normalizeAngle(angle) * Math.PI / 180;
  return {c: snap(Math.cos(radians)), s: snap(Math.sin(radians))};
}
export function localToImage(rect: RotatedRoiRectangle, x: number, y: number): Point {
  const {c, s} = rotation(rect.angle_degrees);
  return {x: rect.center_x + c * x - s * y, y: rect.center_y + s * x + c * y};
}
export function roiCorners(rectangle: RoiRectangle): Point[] {
  const rect = orientedRoi(rectangle);
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => localToImage(rect, x * rect.width / 2, y * rect.height / 2));
}
export function containsPixel(rect: RotatedRoiRectangle, x: number, y: number) {
  const {c, s} = rotation(rect.angle_degrees);
  const dx = x + .5 - rect.center_x, dy = y + .5 - rect.center_y;
  const snap = (v: number, half: number) => Math.abs(v - half) < epsilon ? half : Math.abs(v + half) < epsilon ? -half : v;
  const u = snap(c * dx + s * dy, rect.width / 2), v = snap(-s * dx + c * dy, rect.height / 2);
  return u >= -rect.width / 2 && u < rect.width / 2 && v >= -rect.height / 2 && v < rect.height / 2;
}
export function roiValidation(rect: RotatedRoiRectangle, width: number, height: number): string | null {
  if (rect.version !== 2 || ![rect.center_x, rect.center_y, rect.width, rect.height, rect.angle_degrees].every(Number.isFinite) || !Number.isInteger(rect.width) || !Number.isInteger(rect.height) || rect.width < 1 || rect.height < 1) return 'Bitte eine gültige ROI mit mindestens 1 × 1 Pixel einstellen.';
  const corners = roiCorners(rect);
  if (corners.some(p => p.x < -epsilon || p.y < -epsilon || p.x > width + epsilon || p.y > height + epsilon)) return 'Die gedrehte ROI muss vollständig innerhalb des Bildes liegen. Bitte verkleinern, verschieben oder den Winkel ändern.';
  const minX = Math.max(0, Math.floor(Math.min(...corners.map(p => p.x))));
  const minY = Math.max(0, Math.floor(Math.min(...corners.map(p => p.y))));
  const maxX = Math.min(width, Math.ceil(Math.max(...corners.map(p => p.x))));
  const maxY = Math.min(height, Math.ceil(Math.max(...corners.map(p => p.y))));
  for (let y = minY; y < maxY; y++) for (let x = minX; x < maxX; x++) if (containsPixel(rect, x, y)) return null;
  return 'Die ROI enthält keinen Originalpixelmittelpunkt. Bitte vergrößern oder verschieben.';
}
export function dragRectangle(rect: RotatedRoiRectangle, mode: Exclude<DragMode, 'rotate'>, dx: number, dy: number): RotatedRoiRectangle {
  if (mode === 'move') return {...rect, center_x: rect.center_x + Math.round(dx), center_y: rect.center_y + Math.round(dy)};
  const {c, s} = rotation(rect.angle_degrees);
  const horizontal = mode.endsWith('l') ? -1 : 1, vertical = mode.startsWith('t') ? -1 : 1;
  const width = Math.max(1, Math.round(rect.width + horizontal * (c * dx + s * dy)));
  const height = Math.max(1, Math.round(rect.height + vertical * (-s * dx + c * dy)));
  const shift = localToImage(rect, horizontal * (width - rect.width) / 2, vertical * (height - rect.height) / 2);
  return {...rect, center_x: shift.x, center_y: shift.y, width, height};
}
export function rotateRectangle(rect: RotatedRoiRectangle, start: Point, end: Point): RotatedRoiRectangle {
  if (Math.hypot(end.x - rect.center_x, end.y - rect.center_y) < epsilon) return rect;
  const delta = Math.atan2(end.y - rect.center_y, end.x - rect.center_x) - Math.atan2(start.y - rect.center_y, start.x - rect.center_x);
  return {...rect, angle_degrees: normalizeAngle(Math.round((rect.angle_degrees + delta * 180 / Math.PI) * 10) / 10)};
}
export function roiCoordinates(rectangle: RoiRectangle) {
  const rect = orientedRoi(rectangle);
  const fmt = (n: number) => String(Number(n.toFixed(2)));
  return `Ecken (Pixelgrenzen): ${roiCorners(rect).map(p => `(${fmt(p.x)}, ${fmt(p.y)})`).join(', ')} · Winkel: ${fmt(normalizeAngle(rect.angle_degrees))}° · Ausschnitt: ${rect.width} × ${rect.height} Pixel`;
}
