"""Fixed yaw/calibration, spatially reserved baseline; legacy tasks untouched."""
import numpy as np
import sapien
from .sweepick_block40_headcal import sweepick_block40_headcal
from sweepick_baseline.expert import ExpertMixin
from sweepick_baseline.contract import CROP_MIN, CROP_MAX


class sweepick_block40_act_baseline(ExpertMixin, sweepick_block40_headcal):
    TABLE_CENTER_WORLD_XY=(0.,-.25)
    CROP_MIN=np.array(CROP_MIN)
    CROP_MAX=np.array(CROP_MAX)

    def setup_demo(self, **kwargs):
        self.baseline_point=kwargs['baseline_point']
        self.BLOCK_ROBOT_XY=np.asarray(self.baseline_point['xy'],float)
        self.visual_seed=int(kwargs['seed'])
        self.visual_rng=np.random.default_rng(self.visual_seed)
        if (float(kwargs.get('sweepick_head_translation_dr_m',0))!=0 or
            float(kwargs.get('sweepick_head_rotation_dr_deg',0))!=0):
            raise ValueError('Calibration randomization is forbidden in this milestone')
        dr=kwargs['domain_randomization']
        if dr.get('random_head_camera_dis',0)!=0 or dr.get('random_table_height',0)!=0:
            raise ValueError('Fixed camera and table geometry required')
        super().setup_demo(**kwargs)
        self.save_freq=None

    def load_actors(self):
        super().load_actors()
        # Pure appearance DR; fixed size, mass, initial orientation and pose.
        self.visual_colors={}
        for name,entity in [('block',self.object.actor),('floor',self.table),('background',self.wall.actor)]:
            rgb=self.visual_rng.uniform(.12,.9,3).tolist()
            self.visual_colors[name]=rgb
            for component in entity.get_components():
                if isinstance(component,sapien.render.RenderBodyComponent):
                    for shape in component.render_shapes:shape.material.base_color=[*rgb,1.]
        self.scene.update_render()
