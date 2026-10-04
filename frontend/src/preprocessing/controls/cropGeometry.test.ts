import { describe, expect, it } from 'vitest';
import { cropRectangle, fullImageCrop } from './cropGeometry';
import { roiValidation, dragRectangle } from '../../imageGeometry/rectangleGeometry';

describe('preprocessing crop geometry', () => {
  it('initializes a full-image rectangle and rejects rotation without margin', () => {
    const full = fullImageCrop(100, 80);
    expect(full).toEqual({version:2,center_x:50,center_y:40,width:100,height:80,angle_degrees:0});
    expect(roiValidation(full,100,80)).toBeNull();
    expect(roiValidation({...full,angle_degrees:30},100,80)).toContain('innerhalb');
  });
  it('converts the effective legacy region without changing its stored config', () => {
    const old = {x:95.8,y:70.3,width:20,height:30};
    expect(cropRectangle(old,{width:100,height:80})).toEqual({version:2,center_x:97.5,center_y:75,width:5,height:10,angle_degrees:0});
    expect(old).toEqual({x:95.8,y:70.3,width:20,height:30});
    expect(cropRectangle({},{width:200,height:200}).width).toBe(128);
  });
  it('uses only new geometry when both formats are present and preserves rotation on resize', () => {
    const roi = {...fullImageCrop(20,20),center_x:40,center_y:40,angle_degrees:30};
    expect(cropRectangle({roi,x:999,width:999},{width:80,height:80})).toEqual(roi);
    expect(dragRectangle(roi,'tr',4,3).angle_degrees).toBe(30);
  });
  it('does not silently turn malformed geometry into a valid old crop', () => {
    for (const roi of [null, {}, {version:3}, {...fullImageCrop(2,2),angle_degrees:NaN}]) {
      expect(roiValidation(cropRectangle({roi},{width:10,height:10}),10,10)).not.toBeNull();
    }
  });
});
