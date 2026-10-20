"""Simulator-independent mixin; used only by the new baseline environment."""
import math
import numpy as np
from .contract import STOW, PHASES, validate_path
from .capture import robot_vectors


class StepObserver:
    """Wrap this task's scene only; no SAPIEN class/global monkey patch."""
    def __init__(self, scene, task, writer=None, hz=10):
        self.scene, self.task, self.writer = scene, task, writer
        self.dt=float(scene.get_timestep());self.tick=0
        self.stride=round(1/(hz*self.dt))
        if self.stride<1 or not math.isclose(self.stride*self.dt,1/hz,abs_tol=1e-8):
            raise ValueError('Physics timestep must divide policy period exactly')

    def __getattr__(self, name): return getattr(self.scene,name)

    def step(self):
        self.task.guard_current_commands()
        if self.writer is not None:
            q,a,g=robot_vectors(self.task.robot)
            self.writer.control_frame(self.tick*self.dt,self.task.phase,q,a,g)
        if self.writer is not None and self.tick % self.stride == 0:
            self.writer.frame(self.tick*self.dt,self.task.phase,self.task.get_obs(),q,a,g,
                              self.task.object.get_pose().p)
        self.scene.step();self.tick+=1
        self.task.after_physics_step()


class ExpertMixin:
    def arm_limits(self, arm):
        return np.asarray([j.get_limits()[0] for j in getattr(self.robot,arm+'_arm_joints')])

    def guard_current_commands(self):
        for arm in STOW:
            joints=getattr(self.robot,arm+'_arm_joints')
            validate_path([[float(j.get_drive_target()[0]) for j in joints]],arm,self.arm_limits(arm))

    def take_dense_action(self, control_seq, save_freq=-1):
        # Validate entire planned path before its first command (incl. Stow).
        for arm in STOW:
            path=control_seq.get(arm+'_arm')
            if path is not None:
                validate_path(path['position'],arm,self.arm_limits(arm))
        return super().take_dense_action(control_seq,save_freq=None)

    def in_stow(self):
        q,_,_=robot_vectors(self.robot)
        arm=self.baseline_point['arm'];off=0 if arm=='left' else 6
        return bool(np.max(np.abs(q[off:off+5]-STOW[arm]))<=.025)

    def grasp_contact(self):
        # Compare actual entities, not duplicated left/right link names.
        arm=self.baseline_point['arm']
        links=[g[0].child_link.entity for g in getattr(self.robot,arm+'_gripper')]
        fixed=getattr(self.robot,arm+'_fix_gripper_name')
        links += [l.entity for l in getattr(self.robot,arm+'_entity').get_links() if l.get_name() in fixed]
        block=self.object.actor
        for c in self.scene.get_contacts():
            a,b=(body.entity for body in c.bodies)
            if len(c.points) and ((a==block and b in links) or (b==block and a in links)):
                return True
        return False

    def after_physics_step(self):
        if self.phase=='hold':
            self.metrics['hold_steps']+=1
            valid=(self.in_stow() and float(self.object.get_pose().p[2])-self.initial_z>=.03
                   and self.grasp_contact())
            self.metrics['hold_valid']=self.metrics['hold_valid'] and bool(valid)

    def begin_baseline(self, writer=None):
        self.phase='home';self.success=False
        self.metrics={'lift_success':False,'stow_reached':False,'hold_steps':0,'hold_valid':True,
                      'failure_stage':'home','success':False}
        self.initial_z=float(self.object.get_pose().p[2])
        self.raw_scene=self.scene
        self.scene=StepObserver(self.raw_scene,self,writer)

    def set_phase(self, phase):
        if phase not in PHASES:raise ValueError(phase)
        self.phase=phase;self.metrics['failure_stage']=phase

    def settle(self, seconds):
        for _ in range(math.ceil(seconds/self.scene.dt)):
            self.robot._entity_qf(self.robot.left_entity)
            self.robot._entity_qf(self.robot.right_entity)
            self.scene.step()

    def finish_hold(self):
        self.set_phase('hold');self.settle(2.)
        self.success=bool(self.metrics['lift_success'] and self.metrics['stow_reached'] and
            self.metrics['hold_valid'] and self.metrics['hold_steps']*self.scene.dt>=2.)
        self.metrics.update(success=self.success,hold_seconds=self.metrics['hold_steps']*self.scene.dt)
        if self.success:self.metrics['failure_stage']=''
        return self.metrics

    def play_once(self):
        from envs.utils import ArmTag
        arm=ArmTag(self.baseline_point['arm'])
        # Home has already been set by Base_Task using measured config.
        self.settle(.1)
        q,_,_=robot_vectors(self.robot)
        for side,offset in [('left',0),('right',6)]:
            if np.max(np.abs(q[offset:offset+5]-getattr(self.robot,side+'_homestate')))>.025:
                return self.metrics
        self.set_phase('pregrasp')
        pre,grasp=self.choose_grasp_pose(self.object,arm_tag=arm,pre_dis=.10,target_dis=0.,contact_point_id=None)
        if pre is None or grasp is None:return self.metrics
        self.move(self.move_to_pose(arm_tag=arm,target_pose=pre))
        if not self.plan_success:return self.metrics
        self.set_phase('grasp');self.move(self.move_to_pose(arm_tag=arm,target_pose=grasp))
        if not self.plan_success:return self.metrics
        self.set_phase('close');self.move(self.close_gripper(arm_tag=arm,pos=0.0))
        if not self.plan_success:return self.metrics
        self.set_phase('lift');before=float(self.object.get_pose().p[2])
        self.move(self.move_by_displacement(arm_tag=arm,z=.05,move_axis='world'))
        self.metrics['lift_delta_z_m']=float(self.object.get_pose().p[2])-before
        self.metrics['lift_success']=bool(self.plan_success and self.metrics['lift_delta_z_m']>=.03)
        if not self.metrics['lift_success']:return self.metrics
        self.set_phase('stow')
        # This joint-space path goes through take_dense_action's full-path guard.
        self.move_to_joint_state(arm_tag=arm,target_joint_pos=STOW[str(arm)])
        if not self.plan_success:return self.metrics
        self.settle(.5)
        self.metrics['stow_reached']=self.in_stow()
        if not self.metrics['stow_reached']:return self.metrics
        # No new gripper command during Stow/Hold, no mobile-base motion.
        return self.finish_hold()

    def check_success(self):return bool(getattr(self,'success',False))
