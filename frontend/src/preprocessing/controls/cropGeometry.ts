import { orientedRoi, type RotatedRoiRectangle } from '../../imageGeometry/rectangleGeometry';

export function cropRectangle(config: Record<string, unknown>, image: {width: number; height: number}): RotatedRoiRectangle {
  if ('roi' in config) {
    const roi = config.roi as Record<string, unknown> | null;
    const number = (key: string) => typeof roi?.[key] === 'number' ? roi[key] as number : NaN;
    return {version: roi?.version as 2, center_x: number('center_x'), center_y: number('center_y'),
      width: number('width'), height: number('height'), angle_degrees: roi?.angle_degrees === undefined ? 0 : number('angle_degrees')};
  }
  // Match the legacy runtime, including truncation, defaults and edge clipping.
  const x = Math.max(0, Math.min(Math.trunc(Number(config.x ?? 0)), image.width - 1));
  const y = Math.max(0, Math.min(Math.trunc(Number(config.y ?? 0)), image.height - 1));
  const width = Math.min(image.width - x, Math.max(1, Math.trunc(Number(config.width ?? 128))));
  const height = Math.min(image.height - y, Math.max(1, Math.trunc(Number(config.height ?? 128))));
  return orientedRoi({x, y, width, height});
}

export function fullImageCrop(width: number, height: number): RotatedRoiRectangle {
  return orientedRoi({x: 0, y: 0, width, height});
}
