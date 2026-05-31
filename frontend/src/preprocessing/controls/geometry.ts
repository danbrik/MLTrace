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
