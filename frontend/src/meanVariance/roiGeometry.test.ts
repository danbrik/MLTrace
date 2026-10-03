import { describe, expect, it } from 'vitest';
import { dragRectangle, roiCoordinates } from './roiGeometry';
const rectangle = {x: 10, y: 20, width: 30, height: 40};
describe('ROI pixel selection', () => {
  it('resizes all four corners without moving the opposite corner', () => {
    expect(dragRectangle(rectangle, 'tl', 2, 3, 100, 100)).toEqual({x:12,y:23,width:28,height:37});
    expect(dragRectangle(rectangle, 'tr', 2, 3, 100, 100)).toEqual({x:10,y:23,width:32,height:37});
    expect(dragRectangle(rectangle, 'bl', 2, 3, 100, 100)).toEqual({x:12,y:20,width:28,height:43});
    expect(dragRectangle(rectangle, 'br', 2, 3, 100, 100)).toEqual({x:10,y:20,width:32,height:43});
  });
  it('preserves size when moving and clamps to all boundaries', () => {
    expect(dragRectangle(rectangle, 'move', -99, -99, 100, 100)).toEqual({...rectangle,x:0,y:0});
    expect(dragRectangle(rectangle, 'move', 999, 999, 100, 100)).toEqual({...rectangle,x:70,y:60});
  });
  it('keeps at least one pixel and permits full-image boundary coordinates', () => {
    expect(dragRectangle(rectangle, 'tl', 999, 999, 100, 100)).toEqual({x:39,y:59,width:1,height:1});
    const full = {x:0,y:0,width:100,height:100};
    expect(dragRectangle(full, 'br', 100, 100, 100, 100)).toEqual(full);
    expect(dragRectangle(full, 'br', -100, -100, 100, 100)).toEqual({x:0,y:0,width:1,height:1});
  });
  it('rounds fractional screen positions and reports the same crop boundaries', () => {
    expect(dragRectangle(rectangle, 'move', 1.6, -1.4, 100, 100)).toEqual({...rectangle,x:12,y:19});
    expect(roiCoordinates(rectangle)).toContain('(40, 60)');
    expect(roiCoordinates(rectangle)).toContain('30 × 40 Pixel');
  });
});
