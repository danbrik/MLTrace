import type { PointerEvent } from 'react';

import type { PreprocessingPreviewImage } from '../../types';

export type Point = { x: number; y: number };

export type CropMode = 'move' | 'tl' | 'br';

export type CropDrag = {
  mode: CropMode;
  startX: number;
  startY: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function defaultPoints(width: number, height: number): Point[] {
  const marginX = Math.round(width * 0.2);
  const marginY = Math.round(height * 0.2);
  return [
    { x: marginX, y: marginY },
    { x: Math.max(marginX + 1, width - 1 - marginX), y: marginY },
    { x: Math.max(marginX + 1, width - 1 - marginX), y: Math.max(marginY + 1, height - 1 - marginY) },
    { x: marginX, y: Math.max(marginY + 1, height - 1 - marginY) },
  ];
}

export function orderQuadPoints(points: Point[]): Point[] {
  if (points.length !== 4) return points;
  const center = {
    x: points.reduce((sum, point) => sum + point.x, 0) / 4,
    y: points.reduce((sum, point) => sum + point.y, 0) / 4,
  };
  const ordered = [...points].sort(
    (a, b) => Math.atan2(a.y - center.y, a.x - center.x) - Math.atan2(b.y - center.y, b.x - center.x),
  );
  const start = ordered.reduce((best, point, index) => {
    const candidate = point.x + point.y;
    const current = ordered[best].x + ordered[best].y;
    return candidate < current || (candidate === current && point.y < ordered[best].y) ? index : best;
  }, 0);
  return [...ordered.slice(start), ...ordered.slice(0, start)];
}

export function rectifiedQuadSize(points: Point[]): { width: number; height: number } | null {
  if (points.length !== 4) return null;
  const [topLeft, topRight, bottomRight, bottomLeft] = orderQuadPoints(points);
  if (new Set([topLeft, topRight, bottomRight, bottomLeft].map((point) => `${point.x}:${point.y}`)).size !== 4) {
    return null;
  }
  const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
  return {
    width: Math.max(1, Math.round(Math.max(distance(topLeft, topRight), distance(bottomLeft, bottomRight)))),
    height: Math.max(1, Math.round(Math.max(distance(topLeft, bottomLeft), distance(topRight, bottomRight)))),
  };
}

// Maps a pointer position to integer pixel coordinates in the given image's space.
export function pointFromEvent(event: PointerEvent<HTMLDivElement>, image: PreprocessingPreviewImage): Point {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: Math.round(clamp((event.clientX - rect.left) / rect.width, 0, 1) * image.width),
    y: Math.round(clamp((event.clientY - rect.top) / rect.height, 0, 1) * image.height),
  };
}

// Clamps a crop config to the bounds of the input image it operates on.
export function cropRectFromConfig(
  config: Record<string, unknown>,
  image: PreprocessingPreviewImage,
): { x: number; y: number; width: number; height: number } {
  const x = clamp(Number(config.x ?? 0), 0, image.width - 1);
  const y = clamp(Number(config.y ?? 0), 0, image.height - 1);
  const width = clamp(Number(config.width ?? image.width), 1, image.width - x);
  const height = clamp(Number(config.height ?? image.height), 1, image.height - y);
  return { x, y, width, height };
}
