import logging
from copy import copy
from enum import Enum

import numpy as np
from pydrake.all import (AbstractValue, AddMultibodyPlantSceneGraph, AngleAxis,
                         Concatenate, DiagramBuilder, InputPortIndex,
                         LeafSystem, MeshcatVisualizer, Parser,
                         PiecewisePolynomial, PiecewisePose, PointCloud,
                         PortSwitch, RandomGenerator, RigidTransform,
                         RollPitchYaw, Simulator, StartMeshcat,
                         UniformlyRandomRotationMatrix)

from manipulation import FindResource, running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.pick import (MakeGripperCommandTrajectory, MakeGripperFrames,
                               MakeGripperPoseTrajectory)
from manipulation.scenarios import (AddIiwaDifferentialIK, AddPackagePaths,
                                    MakeManipulationStation, ycb)

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('Differential IK')

logging.getLogger("drake").addFilter(NoDiffIKWarnings())

from pydrake.all import (
  Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
)

# Start the visualizer.
meshcat = StartMeshcat()

FILENAME_MODEL = "clutter_clearing_two_arms.dmd.yaml"
INNER_FILENAME_MODEL = "clutter_planning_two_arms.dmd.yaml"

rs = np.random.RandomState()  # this is for python
generator = RandomGenerator(rs.randint(1000))  # this is for c++


# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    AddPackagePaths(parser)
    f = open("manipulation/"+INNER_FILENAME_MODEL, "r")

    model_directives = f.read()

    parser.AddModelsFromString(
        model_directives, 'dmd.yaml')
    plant.Finalize()
    return builder.Build()


# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class GraspSelector(LeafSystem):
    def __init__(self, plant, bin_instance, camera_body_indices, arm_index):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        port = self.DeclareAbstractOutputPort(
            "grasp_selection", lambda: AbstractValue.Make(
                (np.inf, RigidTransform())), self.SelectGrasp)
        port.disable_caching_by_default()

        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply([-.22 + 0.025 + margin, -.29 + 0.025 + margin, 0.015 + margin])
        b = X_B.multiply([.22 - 0.1 - margin, .29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        self._internal_model = make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng()
        self._camera_body_indices = camera_body_indices
        self.arm_index = arm_index
        _internal_model_plant = self._internal_model.GetSubsystemByName("plant")
        if arm_index == 1:
            self.wsg_body_idx = _internal_model_plant.GetBodyByName("body",
                                                                    _internal_model_plant.GetModelInstanceByName(
                                                                        "wsg")).index()
        else:
            self.wsg_body_idx = _internal_model_plant.GetBodyByName("body",
                                                                    _internal_model_plant.GetModelInstanceByName(
                                                                        "wsg2")).index()

    def SelectGrasp(self, context, output):
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.

        for i in range(100 if running_as_notebook else 2):
            print("arm index", self.arm_index)
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model, self._internal_model_context,
                down_sampled_pcd, self._rng, wsg_body_index=self.wsg_body_idx)
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            X_WG = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                                  [0.5, 0, 0.22])
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))


class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING_FROM_X_BIN = 2
    PICKING_FROM_Y_BIN = 3
    GO_HOME = 4


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        iiwa1_model_instance_index = plant.GetModelInstanceByName("iiwa1")
        iiwa2_model_instance_index = plant.GetModelInstanceByName("iiwa2")
        wsg1_model_instance_index = plant.GetModelInstanceByName("wsg")
        wsg2_model_instance_index = plant.GetModelInstanceByName("wsg2")

        self._gripper_body_index_1 = plant.GetBodyByName("body", wsg1_model_instance_index).index()
        self._gripper_body_index_2 = plant.GetBodyByName("body", wsg2_model_instance_index).index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "x_bin_grasp", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._x_bin_grasp_index2 = self.DeclareAbstractInputPort(
            "x_bin_grasp2", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._y_bin_grasp_index = self.DeclareAbstractInputPort(
            "y_bin_grasp", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._y_bin_grasp_index2 = self.DeclareAbstractInputPort(
            "y_bin_grasp2", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state",
                                                            2).get_index()
        self._wsg_state_index2 = self.DeclareVectorInputPort("wsg_state2",
                                                             2).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE))
        self._mode_index2 = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE))
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_X_G_index_2 = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self._traj_wsg_index_2 = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self._times_index = self.DeclareAbstractState(AbstractValue.Make(
            {"initial": 0.0}))
        self._times_index2 = self.DeclareAbstractState(AbstractValue.Make(
            {"initial": 0.0}))
        self._attempts_index = self.DeclareDiscreteState(1)
        self._attempts_index2 = self.DeclareDiscreteState(1)

        self.DeclareAbstractOutputPort(
            "X_WG2", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose2)
        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position2", 1, self.CalcWsgPosition2)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # For GoHome mode.
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa1_position", num_positions).get_index()
        self._iiwa2_position_index = self.DeclareVectorInputPort(
            "iiwa2_position", num_positions).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode2", lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode2)
        self.DeclareAbstractOutputPort(
            "control_mode", lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode)
        self.DeclareAbstractOutputPort(
            "reset_diff_ik", lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset)
        self.DeclareAbstractOutputPort(
            "reset_diff_ik2", lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset2)

        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._q0_index2 = self.DeclareDiscreteState(num_positions)  # for q0

        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self._traj_q_index_2 = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self.DeclareVectorOutputPort("iiwa1_position_command", num_positions,
                                     self.CalcIiwaPosition)
        self.DeclareVectorOutputPort("iiwa2_position_command", num_positions,
                                     self.CalcIiwaPosition2)
        # self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize2)

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update2)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(
            self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(int(
                self._traj_q_index)).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object.  Time to replan.
        if (current_time > times["postpick"] and
                current_time < times["preplace"]):
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
            if wsg_state[0] < 0.01:
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index)).get_mutable_value()
                if attempts[0] > 5:
                    # If I've failed 5 times in a row, then switch bins.
                    print(
                        "Switching to the other bin after 5 consecutive failed attempts"
                    )
                    attempts[0] = 0
                    if mode == PlannerState.PICKING_FROM_X_BIN:
                        state.get_mutable_abstract_state(int(
                            self._mode_index)).set_value(
                            PlannerState.PICKING_FROM_Y_BIN)
                    else:
                        state.get_mutable_abstract_state(int(
                            self._mode_index)).set_value(
                            PlannerState.PICKING_FROM_X_BIN)
                    self.Plan(context, state)  # TODO: changed to Plan2
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(int(
                    self._mode_index)).set_value(
                    PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                times = {"initial": current_time}
                state.get_mutable_abstract_state(int(
                    self._times_index)).set_value(times)
                X_G = self.get_input_port(0).Eval(context)[int(
                    self._gripper_body_index_1)]
                state.get_mutable_abstract_state(int(
                    self._traj_X_G_index)).set_value(PiecewisePose.MakeLinear([current_time, np.inf], [X_G, X_G]))
                return

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index)).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[int(
            self._gripper_body_index_1)]
        # if current_time > 10 and current_time < 12:
        #    self.GoHome(context, state)
        #    return
        if np.linalg.norm(
                traj_X_G.GetPose(current_time).translation()
                - X_G.translation()) > 0.2:
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome(context, state)
            return

    def Update2(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index2)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(
            self._times_index2)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan2(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(int(
                self._traj_q_index_2)).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan2(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object.  Time to replan.
        if (current_time > times["postpick"] and
                current_time < times["preplace"]):
            wsg_state = self.get_input_port(self._wsg_state_index2).Eval(context)
            if wsg_state[0] < 0.01:
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index2)).get_mutable_value()
                if attempts[0] > 5:
                    # If I've failed 5 times in a row, then switch bins.
                    print(
                        "Switching to the other bin after 5 consecutive failed attempts"
                    )
                    attempts[0] = 0
                    if mode == PlannerState.PICKING_FROM_X_BIN:
                        state.get_mutable_abstract_state(int(
                            self._mode_index2)).set_value(
                            PlannerState.PICKING_FROM_Y_BIN)
                    else:
                        state.get_mutable_abstract_state(int(
                            self._mode_index2)).set_value(
                            PlannerState.PICKING_FROM_X_BIN)
                    self.Plan2(context, state)
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(int(
                    self._mode_index2)).set_value(
                    PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                times = {"initial": current_time}
                state.get_mutable_abstract_state(int(
                    self._times_index2)).set_value(times)
                X_G = self.get_input_port(0).Eval(context)[int(
                    self._gripper_body_index_2)]
                state.get_mutable_abstract_state(int(
                    self._traj_X_G_index_2)).set_value(PiecewisePose.MakeLinear([current_time, np.inf], [X_G, X_G]))
                return

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index_2)).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan2(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[int(
            self._gripper_body_index_2)]
        # if current_time > 10 and current_time < 12:
        #    self.GoHome(context, state)
        #    return
        if np.linalg.norm(
                traj_X_G.GetPose(current_time).translation()
                - X_G.translation()) > 0.2:
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome2(context, state)
            return

    def GoHome2(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(
            self._mode_index2)).set_value(
            PlannerState.GO_HOME)
        q = self.get_input_port(self._iiwa2_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index2).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T)
        state.get_mutable_abstract_state(int(
            self._traj_q_index_2)).set_value(q_traj)

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(
            self._mode_index)).set_value(
            PlannerState.GO_HOME)
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T)
        state.get_mutable_abstract_state(int(
            self._traj_q_index)).set_value(q_traj)

    def Plan(self, context, state):

        mode = copy(
            state.get_mutable_abstract_state(int(self._mode_index)).get_value())

        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index_1)]
        }

        cost = np.inf
        for i in range(5):
            if mode == PlannerState.PICKING_FROM_Y_BIN:
                cost, X_G["pick"] = self.get_input_port(
                    self._y_bin_grasp_index).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_X_BIN
            else:
                cost, X_G["pick"] = self.get_input_port(
                    self._x_bin_grasp_index).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_Y_BIN
                else:
                    mode = PlannerState.PICKING_FROM_X_BIN

            if not np.isinf(cost):
                break

        assert not np.isinf(
            cost), "Could not find a valid grasp in either bin after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_FROM_X_BIN:
            # Place in Y bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [rs.uniform(-.25, .15),
                 rs.uniform(-.6, -.4), .3])
        else:
            # Place in X bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                [rs.uniform(.35, .65),
                 rs.uniform(-.12, .28), .3])

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(
            self._times_index)).set_value(times)

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_G["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index)).set_value(traj_wsg_command)

    def Plan2(self, context, state):

        mode = copy(
            state.get_mutable_abstract_state(int(self._mode_index2)).get_value())

        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index_2)]
        }

        cost = np.inf
        for i in range(5):
            if mode == PlannerState.PICKING_FROM_Y_BIN:
                cost, X_G["pick"] = self.get_input_port(
                    self._y_bin_grasp_index2).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_X_BIN
            else:
                cost, X_G["pick"] = self.get_input_port(
                    self._x_bin_grasp_index2).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_Y_BIN
                else:
                    mode = PlannerState.PICKING_FROM_X_BIN

            if not np.isinf(cost):
                break
        # X_G['pick'] = X_G['pick']@RigidTransform(RotationMatrix.MakeXRotation(0), np.array([-0.5,0.5,0]))
        print(X_G['pick'])
        assert not np.isinf(
            cost), "Could not find a valid grasp in either bin after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index2)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        # TODO(marcel): issue hardcoded bin positions right?
        if mode == PlannerState.PICKING_FROM_X_BIN:
            # Place in Y bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [rs.uniform(-.25, .15),
                 rs.uniform(-.6, -.4), .3])
        else:
            # Place in X bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                [rs.uniform(.35, .65),
                 rs.uniform(-.12, .28), .3])

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(
            self._times_index2)).set_value(times)

        if True:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_G["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index_2)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index_2)).set_value(traj_wsg_command)

    """
    def start_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()

    def start_time2(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time2(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()
    """

    def CalcGripperPose(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index)).get_value()
        if (traj_X_G.get_number_of_segments() > 0 and
                traj_X_G.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(
                    self._traj_X_G_index)).get_value().GetPose(
                    context.get_time()))
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(self.get_input_port(0).Eval(context)
                         [int(self._gripper_body_index_1)])

    def CalcGripperPose2(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index2)).get_value()

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index_2)).get_value()
        if (traj_X_G.get_number_of_segments() > 0 and
                traj_X_G.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(
                    self._traj_X_G_index_2)).get_value().GetPose(
                    context.get_time()))
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(self.get_input_port(0).Eval(context)
                         [int(self._gripper_body_index_2)])

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(
            self._traj_wsg_index)).get_value()
        if (traj_wsg.get_number_of_segments() > 0 and
                traj_wsg.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcWsgPosition2(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index2)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(
            self._traj_wsg_index_2)).get_value()
        if (traj_wsg.get_number_of_segments() > 0 and
                traj_wsg.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode2(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index2)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def CalcDiffIKReset2(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index2)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context))

    def Initialize2(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index2),
            self.get_input_port(int(self._iiwa2_position_index)).Eval(context))

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(
            self._traj_q_index)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))

    def CalcIiwaPosition2(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(
            self._traj_q_index_2)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))


def clutter_clearing_demo():
    meshcat.Delete()
    builder = DiagramBuilder()
    f = open("manipulation/"+FILENAME_MODEL, "r")

    model_directives = f.read()
    # print(model_directives)

#     for i in range(10 if running_as_notebook else 2):
#         object_num = rs.randint(len(ycb))
#         if "cracker_box" in ycb[object_num]:
#             # skip it. it's just too big!
#             continue
#         model_directives += f"""
# - add_model:
#     name: ycb{i}
# #    file: package://drake/examples/manipulation_station/models/061_foam_brick.sdf
#     file: package://drake/manipulation/models/ycb/sdf/{ycb[object_num]}
# """

    station = builder.AddSystem(
        MakeManipulationStation(model_directives, time_step=0.001))

    plant = station.GetSubsystemByName("plant")

    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin0"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera0"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera1"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera2"))[0]
                      ], arm_index=1))

    y_bin_grasp_selector2 = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin0"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera0"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera1"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera2"))[0]
                      ], arm_index=2))
    builder.Connect(station.GetOutputPort("camera0_point_cloud"),
                    y_bin_grasp_selector.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera1_point_cloud"),
                    y_bin_grasp_selector.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera2_point_cloud"),
                    y_bin_grasp_selector.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    y_bin_grasp_selector.GetInputPort("body_poses"))

    builder.Connect(station.GetOutputPort("camera0_point_cloud"),
                    y_bin_grasp_selector2.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera1_point_cloud"),
                    y_bin_grasp_selector2.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera2_point_cloud"),
                    y_bin_grasp_selector2.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    y_bin_grasp_selector2.GetInputPort("body_poses"))

    x_bin_grasp_selector = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin1"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera3"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera4"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera5"))[0]
                      ], arm_index=1))

    x_bin_grasp_selector2 = builder.AddSystem(
        GraspSelector(plant,
                      plant.GetModelInstanceByName("bin1"),
                      camera_body_indices=[
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera3"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera4"))[0],
                          plant.GetBodyIndices(
                              plant.GetModelInstanceByName("camera5"))[0]
                      ], arm_index=2))

    builder.Connect(station.GetOutputPort("camera3_point_cloud"),
                    x_bin_grasp_selector.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera4_point_cloud"),
                    x_bin_grasp_selector.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera5_point_cloud"),
                    x_bin_grasp_selector.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    x_bin_grasp_selector.GetInputPort("body_poses"))

    builder.Connect(station.GetOutputPort("camera3_point_cloud"),
                    x_bin_grasp_selector2.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera4_point_cloud"),
                    x_bin_grasp_selector2.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera5_point_cloud"),
                    x_bin_grasp_selector2.get_input_port(2))
    builder.Connect(station.GetOutputPort("body_poses"),
                    x_bin_grasp_selector2.GetInputPort("body_poses"))

    planner = builder.AddSystem(Planner(plant))
    builder.Connect(station.GetOutputPort("body_poses"),
                    planner.GetInputPort("body_poses"))
    builder.Connect(x_bin_grasp_selector.get_output_port(),
                    planner.GetInputPort("x_bin_grasp"))
    builder.Connect(x_bin_grasp_selector2.get_output_port(),
                    planner.GetInputPort("x_bin_grasp2"))
    builder.Connect(y_bin_grasp_selector.get_output_port(),
                    planner.GetInputPort("y_bin_grasp"))
    builder.Connect(y_bin_grasp_selector2.get_output_port(),
                    planner.GetInputPort("y_bin_grasp2"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    planner.GetInputPort("wsg_state"))
    builder.Connect(station.GetOutputPort("wsg2_state_measured"),
                    planner.GetInputPort("wsg_state2"))
    builder.Connect(station.GetOutputPort("iiwa1_position_measured"),
                    planner.GetInputPort("iiwa1_position"))
    builder.Connect(station.GetOutputPort("iiwa2_position_measured"),
                    planner.GetInputPort("iiwa2_position"))

    robot1 = station.GetSubsystemByName(
        "iiwa1_controller").get_multibody_plant_for_control()
    robot2 = station.GetSubsystemByName(
        "iiwa2_controller").get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot1)
    diff_ik2 = AddIiwaDifferentialIK(builder, robot2)

    builder.Connect(planner.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(planner.GetOutputPort("X_WG2"),
                    diff_ik2.get_input_port(0))

    builder.Connect(station.GetOutputPort("iiwa1_state_estimated"),
                    diff_ik.GetInputPort("robot_state"))
    builder.Connect(station.GetOutputPort("iiwa2_state_estimated"),
                    diff_ik2.GetInputPort("robot_state"))

    builder.Connect(planner.GetOutputPort("reset_diff_ik"),
                    diff_ik.GetInputPort("use_robot_state"))
    builder.Connect(planner.GetOutputPort("reset_diff_ik2"),
                    diff_ik2.GetInputPort("use_robot_state"))

    builder.Connect(planner.GetOutputPort("wsg_position2"),
                    station.GetInputPort("wsg2_position"))

    builder.Connect(planner.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(),
                    switch.DeclareInputPort("diff_ik"))
    builder.Connect(planner.GetOutputPort("iiwa1_position_command"),
                    switch.DeclareInputPort("position"))
    builder.Connect(switch.get_output_port(),
                    station.GetInputPort("iiwa1_position"))
    builder.Connect(planner.GetOutputPort("control_mode"),
                    switch.get_port_selector_input_port())

    switch2 = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik2.get_output_port(),
                    switch2.DeclareInputPort("diff_ik2"))
    builder.Connect(planner.GetOutputPort("iiwa2_position_command"),
                    switch2.DeclareInputPort("position2"))
    builder.Connect(switch2.get_output_port(),
                    station.GetInputPort("iiwa2_position"))
    builder.Connect(planner.GetOutputPort("control_mode2"),
                    switch2.get_port_selector_input_port())

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    z = 0.2
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform(
            UniformlyRandomRotationMatrix(generator),
            [rs.uniform(.35, .65), rs.uniform(-.12, .28), z])
        plant.SetFreeBodyPose(plant_context,
                              plant.get_body(body_index),
                              tf)
        z += 0.1

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

    # if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")


clutter_clearing_demo()
