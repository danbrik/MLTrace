import { describe, expect, it } from 'vitest';
import { containsPixel, dragRectangle, orientedRoi, roiCoordinates, roiCorners, roiValidation, rotateRectangle } from './roiGeometry';
const rectangle = orientedRoi({x:10,y:20,width:30,height:40});
describe('rotated ROI pixel selection', () => {
  it('converts old rectangles losslessly and resizes every corner at zero degrees', () => {
    expect(rectangle).toEqual({version:2,center_x:25,center_y:40,width:30,height:40,angle_degrees:0});
    expect(dragRectangle(rectangle,'tl',2,3)).toEqual({...rectangle,center_x:26,center_y:41.5,width:28,height:37});
    expect(dragRectangle(rectangle,'tr',2,3)).toEqual({...rectangle,center_x:26,center_y:41.5,width:32,height:37});
    expect(dragRectangle(rectangle,'bl',2,3)).toEqual({...rectangle,center_x:26,center_y:41.5,width:28,height:43});
    expect(dragRectangle(rectangle,'br',2,3)).toEqual({...rectangle,center_x:26,center_y:41.5,width:32,height:43});
    expect(roiCoordinates(rectangle)).toContain('(40, 60)');
    expect(roiCoordinates(rectangle)).toContain('Winkel: 0°');
  });
  it.each([0,30,-30,45,90,180])('preserves the opposite corner and right angles when resizing at %s degrees', angle => {
    const rect = {...rectangle,angle_degrees:angle};
    const modes = ['tl','tr','br','bl'] as const;
    modes.forEach((mode,index) => {
      const resized = dragRectangle(rect,mode,5,-3);
      const corners = roiCorners(resized), original = roiCorners(rect), opposite=(index+2)%4;
      expect(corners[opposite].x).toBeCloseTo(original[opposite].x,10);
      expect(corners[opposite].y).toBeCloseTo(original[opposite].y,10);
      const a={x:corners[1].x-corners[0].x,y:corners[1].y-corners[0].y}, b={x:corners[2].x-corners[1].x,y:corners[2].y-corners[1].y};
      expect(a.x*b.x+a.y*b.y).toBeCloseTo(0,9);
      expect(resized.angle_degrees).toBe(angle);
      expect(Number.isInteger(resized.width)&&Number.isInteger(resized.height)).toBe(true);
    });
  });
  it('keeps at least one pixel, preserves dimensions on move, and rejects bounds without clipping', () => {
    expect(dragRectangle(rectangle,'tl',999,999)).toMatchObject({width:1,height:1});
    const moved = dragRectangle(rectangle,'move',999,999);
    expect(moved.width).toBe(30);expect(moved.height).toBe(40);
    expect(roiValidation(moved,100,100)).toContain('innerhalb');
    const full=orientedRoi({x:0,y:0,width:100,height:100});
    expect(roiValidation(full,100,100)).toBeNull();
    expect(roiValidation({...full,angle_degrees:30},100,100)).toContain('innerhalb');
  });
  it('rotates clockwise using the handle and leaves center and size unchanged', () => {
    const rotated=rotateRectangle(rectangle,{x:25,y:20},{x:45,y:40});
    expect(rotated).toEqual({...rectangle,angle_degrees:90});
    expect(rotateRectangle(rectangle,{x:25,y:20},{x:25,y:40})).toEqual(rectangle);
  });
  it('matches original pixel centers and detects an empty but geometrically valid diamond', () => {
    const roi={version:2 as const,center_x:1,center_y:1,width:1,height:1,angle_degrees:45};
    expect(roiValidation(roi,2,2)).toContain('keinen Originalpixelmittelpunkt');
    const full=orientedRoi({x:0,y:0,width:3,height:2});
    expect(roiValidation(full,3,2)).toBeNull();
    expect(containsPixel(full,0,0)).toBe(true);expect(containsPixel(full,2,1)).toBe(true);expect(containsPixel(full,3,1)).toBe(false);
    expect(roiValidation({...full,angle_degrees:NaN},3,2)).toContain('gültige ROI');
  });
});
