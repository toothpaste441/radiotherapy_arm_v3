"""NumPy-only tests; no MuJoCo installation needed for these unit tests."""
import numpy as np
from src.path_planning import CollisionChecker, SphereObstacle, rrt_connect, point_segment_distance
from src.trajectory_generation import make_quintic_trajectory


class DummyKinematics:
    joint_limits = np.tile(np.array([-1.5, 1.5]), (6, 1))

    def _chain_state(self, q):
        # Six connected segments along x, with endpoint controlled by q[0:2]
        pivots = np.column_stack((np.linspace(0, 0.5, 6)+q[0],
                                  np.full(6, q[1]), np.zeros(6)))
        return np.array([0.6+q[0], q[1], 0.0]), np.zeros((3,6)), pivots.T


def test_segment_distance():
    assert np.isclose(point_segment_distance([0.5,1,0],[0,0,0],[1,0,0]), 1.0)
    assert np.isclose(point_segment_distance([2,0,0],[0,0,0],[1,0,0]), 1.0)


def test_collision_detects_obstacle_and_limits():
    kin=DummyKinematics()
    checker=CollisionChecker(kin,[SphereObstacle(np.array([0.2,0,0]),0.05)],
                             link_radius=0.01,clearance=0)
    assert not checker.valid(np.zeros(6))
    assert checker.valid(np.array([0,0.5,0,0,0,0]))
    assert not checker.valid(np.array([2,0,0,0,0,0]))


def test_planner_collision_free_path():
    kin=DummyKinematics()
    obs=SphereObstacle(np.array([0.2,0,0]),0.1)
    checker=CollisionChecker(kin,[obs],link_radius=0.015,clearance=0.005)
    start=np.array([0,-0.4,0,0,0,0])
    goal=np.array([0,0.4,0,0,0,0])
    assert not checker.edge_valid(start,goal)
    plan=rrt_connect(start,goal,checker,seed=42,max_iterations=2000)
    assert plan.success
    assert np.allclose(plan.waypoints[0],start)
    assert np.allclose(plan.waypoints[-1],goal)
    assert all(checker.edge_valid(a,b) for a,b in zip(plan.waypoints[:-1],plan.waypoints[1:]))


def test_quintic_endpoints_and_velocity_limit():
    waypoints=np.array([[0,0,0,0,0,0],[0.8,-0.3,0,0,0,0],[-0.2,0.3,0,0,0,0]])
    tr=make_quintic_trajectory(waypoints,max_joint_speed=0.4)
    assert np.allclose(tr.evaluate(0)[0],waypoints[0])
    assert np.allclose(tr.evaluate(tr.duration)[0],waypoints[-1])
    for t in np.linspace(0,tr.duration,201):
        q,qd,qdd=tr.evaluate(t)
        assert np.max(np.abs(qd)) <= 0.4+1e-9
    for t in tr.knots:
        assert np.allclose(tr.evaluate(t)[1],0,atol=1e-8)
        assert np.allclose(tr.evaluate(t)[2],0,atol=1e-8)
