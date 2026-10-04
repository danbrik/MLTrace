import json
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.mean_variance import roi_engine as engine, roi_service, service
from app.mean_variance.roi_geometry import RotatedRectangle, aligned_crop, oriented_roi, selection_mask
from tests.test_reference_image import source
from tests.test_mean_variance import comparison
from tests.test_variance_roi import setup, completed


def rect(angle=0, **changes):
    return RotatedRectangle(version=2, center_x=4, center_y=4, width=4, height=4, angle_degrees=angle, **changes)


def test_zero_degree_exact_legacy_crop_and_mask():
    image = np.arange(63).reshape(7,9)
    old = engine.Rectangle(x=1,y=2,width=5,height=3)
    rotated = oriented_roi(old)
    np.testing.assert_array_equal(aligned_crop(image, rotated), image[2:5,1:6])
    np.testing.assert_array_equal(aligned_crop(image, old), image[2:5,1:6])
    expected = np.zeros_like(image,dtype=bool);expected[2:5,1:6]=True
    np.testing.assert_array_equal(selection_mask(rotated,9,7),expected)
    assert engine.positive_metrics(image,rotated) == engine.positive_metrics(image,old)


@pytest.mark.parametrize('angle,turns', [(0,0),(90,1),(180,2),(-90,-1),(270,-1),(360,0)])
def test_quarter_turn_is_exact_pixel_permutation(angle, turns):
    image=np.arange(64,dtype=np.uint16).reshape(8,8)
    roi=rect(angle)
    np.testing.assert_array_equal(aligned_crop(image,roi),np.rot90(image[2:6,2:6],turns))
    assert selection_mask(roi,8,8).sum()==16


@pytest.mark.parametrize('angle', [30,-30,45,89.5,135])
def test_free_angle_inverse_mapping_and_original_pixel_metrics(angle):
    image=np.arange(64).reshape(8,8)
    roi=rect(angle)
    output=aligned_crop(image,roi)
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    expected=np.empty((4,4),dtype=image.dtype)
    mask=np.zeros((8,8),dtype=bool)
    for y in range(4):
        for x in range(4):
            u,v=x+.5-2,y+.5-2
            sy,sx=np.indices(image.shape,dtype=float)
            distances=(sx+.5-(4+c*u-s*v))**2+(sy+.5-(4+s*u+c*v))**2
            nearest=np.flatnonzero(np.isclose(distances,distances.min(),atol=1e-12,rtol=0))
            expected[y,x]=image.ravel()[nearest[-1]]  # Equal distances: right/down pixel.
    for y in range(8):
        for x in range(8):
            u=c*(x+.5-4)+s*(y+.5-4);v=-s*(x+.5-4)+c*(y+.5-4)
            mask[y,x]=(-2<=u<2 and -2<=v<2)
    np.testing.assert_array_equal(output,expected)
    np.testing.assert_array_equal(selection_mask(roi,8,8),mask)
    assert set(output.ravel()).issubset(set(image.ravel()))
    difference=image.astype(float)-25
    metrics=engine.positive_metrics(difference,roi)
    assert metrics['selected_pixels']==mask.sum()
    assert metrics['area_percent']==mask.sum()/64*100
    assert metrics['positive_roi']==np.maximum(difference,0)[mask].sum()


def test_no_pixel_centers_and_out_of_bounds_are_rejected():
    empty=RotatedRectangle(version=2,center_x=1,center_y=1,width=1,height=1,angle_degrees=45)
    empty.validate_bounds(2,2)
    with pytest.raises(ValueError,match='keinen Originalpixelmittelpunkt'):
        selection_mask(empty,2,2)
    full=RotatedRectangle(version=2,center_x=4,center_y=4,width=8,height=8,angle_degrees=30)
    with pytest.raises(ValueError,match='innerhalb'):
        selection_mask(full,8,8)
    for field,value in [('center_x',float('nan')),('center_y',float('inf')),('angle_degrees',float('inf')),('width',1.5),('height',0),('version',3)]:
        with pytest.raises(ValidationError):
            RotatedRectangle.model_validate({**rect().model_dump(),field:value})


@pytest.mark.parametrize('pairs',[1,6])
def test_rotated_run_saved_export_and_metadata(setup,pairs):
    db,_,_=setup
    run=completed(setup,pairs)
    cfg=engine.RoiConfig(roi={'version':2,'center_x':160.5,'center_y':90.5,'width':60,'height':40,'angle_degrees':30},opacity=.7)
    job=roi_service.enqueue(db,run.id,'evaluate',cfg,wake_scheduler=False)
    roi_service.run_scheduled(job['id']);db.expire_all()
    saved=roi_service.state(db,run.id)['saved']
    assert saved['config']==cfg.model_dump()
    assert saved['result']['roi']==cfg.roi.model_dump()
    assert saved['result']['output_width']==60 and saved['result']['output_height']==40
    assert len(saved['result']['rows'])==pairs
    for name in ('roi_comparison.png','roi_table.png'):
        with Image.open(roi_service.artifact_dir(job['id'])/name) as png:
            metadata=json.loads(png.info['Description'])
            assert metadata['roi']['angle_degrees']==30 and metadata['resampling']=='nearest'
            assert metadata['selected_pixels']==selection_mask(cfg.roi,321,181).sum()
            np.testing.assert_allclose(metadata['roi_corners'],cfg.roi.corners())
    assert service.get_run(db,run.id).status=='finished'


def test_legacy_requests_upgrade_and_existing_results_remain_unchanged(setup):
    from app import models
    db,_,_=setup
    run=completed(setup)
    old=engine.RoiConfig(roi={'x':2,'y':3,'width':4,'height':5})
    job=roi_service.enqueue(db,run.id,'evaluate',old,wake_scheduler=False)
    assert job['config']['roi']==oriented_roi(old.roi).model_dump()
    roi_service.run_scheduled(job['id']);db.expire_all()
    saved=db.get(models.VarianceRoiJob,job['id'])
    # Historical JSON remains readable verbatim; opening does not rewrite artifacts.
    saved.config=old.model_dump();saved.result={**saved.result,'roi':old.roi.model_dump()};db.commit()
    before=(roi_service.artifact_dir(job['id'])/'roi_comparison.png').read_bytes()
    state=roi_service.state(db,run.id)
    assert state['saved']['config']==old.model_dump() and state['saved']['result']['roi']==old.roi.model_dump()
    assert (roi_service.artifact_dir(job['id'])/'roi_comparison.png').read_bytes()==before


def test_renderer_uses_same_pixel_mapping_for_overlay_and_polygon(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    basis=tmp_path/'basis';basis.mkdir()
    np.save(basis/'0_mean.npy',np.arange(64,dtype=float).reshape(8,8))
    np.save(basis/'0_difference.npy',np.arange(64,dtype=float).reshape(8,8)-32)
    engine.finish_basis(basis,{}, {'version':2,'width':8,'height':8,'difference_scale_limit':32,
        'pairs':[{'label':'u1','periods':{},'counts':{}}]},'Dataset','Pipeline')
    cfg=engine.RoiConfig(roi=rect(30))
    original=Figure.savefig
    def inspect(fig,path,*args,**kwargs):
        if path.name=='roi_comparison.png':
            left,right=fig.axes[:2]
            np.testing.assert_array_equal(right.images[0].get_array(),aligned_crop(left.images[0].get_array(),cfg.roi))
            np.testing.assert_allclose(left.patches[0].get_xy()[:4],cfg.roi.corners()-.5)
        return original(fig,path,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',inspect)
    engine.export_roi(basis,tmp_path,cfg)
