import time
import rospy
from operator import xor
from std_msgs.msg import Bool
from raiv_libraries.robotUR import RobotUR
import geometry_msgs.msg as geometry_msgs
from raiv_libraries.srv import ObjectGripped, ObjectGrippedResponse
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
)

"""
Class used to move the robot in all cardinal directions or pick and place an object. 
"""

class Robot_with_vaccum_gripper(RobotUR):
    def __init__(self, gripper_topic='switch_on_off'):
        super().__init__()
        self.previous_z_position = 0  # Used to memorize the z coordinate when performing a go down then go up movement
        self.gripper_topic = gripper_topic  # Gripper topic
        self.gripper_publisher = rospy.Publisher(self.gripper_topic, Bool, queue_size=10)  # Publisher for the gripper topic

    # Action north: positive x
    def small_move_to_north(self, distance):
        self.relative_move(distance, 0, 0)

    # Action south: negative x
    def small_move_to_south(self, distance):
        self.relative_move(-distance, 0, 0)

    # Action east: negative y
    def small_move_to_east(self, distance):
        self.relative_move(0, -distance, 0)

    # Action west: positive y
    def small_move_to_west(self, distance):
        self.relative_move(0, distance, 0)

    def check_if_object_gripped(self):
        try:
            resp = rospy.wait_for_message('object_gripped', Bool)
            return resp.data
        except rospy.ServiceException as e:
            print("Service check_if_object_gripped call failed: %s" % e)


    # Action pick: for pick and place
    def pick(self, pose_for_pick):
        # The robot goes down until the gripper touch something (table or object)
        # Detection provided by /contact topic
        # To be used in recording images for NN training
        self.go_to_pose(pose_for_pick)
        self._send_gripper_message(True, timer=1)   # Vaccum gripper ON
        communication_problem = True
        while communication_problem:  # Infinite loop until the movement is completed
            communication_problem = self._down_movement(movement_duration=4)
        self._back_to_previous_z()  # Back to the original z pose (go up)


    # Function to place the grasped object
    def place(self, pose_for_place):
        self.go_to_pose(pose_for_place)  # Go to place to launch the object


    def release_gripper(self):
        self._send_gripper_message(False)
        #time.sleep(0.5)

    ####################### Privates methods #######################

    def _send_gripper_message(self, msg, timer=2, n_msg=10):
        """
        Function that sends a burst of n messages of the gripper_topic during an indicated time
        :param msg: True or False
        :param time: time in seconds
        :param n_msg: number of messages
        :return:
        """
        time_step = (timer/2)/n_msg
        i=0
        while(i <= n_msg):
            self.gripper_publisher.publish(msg)
            time.sleep(time_step)
            i += 1
        time.sleep(timer/2)

    def _back_to_previous_z(self):
        """
        Function used to go back to the original height once a vertical movement has been performed.
        """
        pose = self.get_current_pose()
        pose.position.z = self.previous_z_position   #0.11 #self.initial_pose.position.z
        self.go_to_position(pose.position)

    def _down_movement(self, movement_duration):
        """
        This function performs the down movement of the pick action.

        Finally, when there is any problems with the communications the movement is stopped and
        communication_problem boolean flag is set to True. It is considered that there is a problem with
        communications when the robot is not receiving any contact messages during 200 milli-seconds (timeout=0.2)

        :param movement_duration: time used to perform the down movement
        :param robot: robot_controller.robot.py object
        :return: communication_problem flag
        """
        contact_ok = rospy.wait_for_message('contact', Bool).data  # We retrieve sensor contact
        communication_problem = False
        if not contact_ok:  # If the robot is already in contact with an object, no movement is performed
            wpose = self.get_current_pose()
            self.previous_z_position = wpose.position.z
            wpose.position.z = 0
            wpose.orientation = self.initial_pose.orientation
            self.trajectory_client.wait_for_server()
            point = CartesianTrajectoryPoint()
            point.pose = wpose
            point.time_from_start = rospy.Duration(movement_duration)
            goal = FollowCartesianTrajectoryGoal()
            goal.trajectory.points.append(point)
            self.trajectory_client.send_goal(goal)
            z = self.current_pose.position.z = 0.2
            while (not contact_ok and z > 0.026) is True:
                z = self.current_pose.position.z
                try:
                    contact_ok = rospy.wait_for_message('contact', Bool, 0.2).data  # We retrieve sensor contact
                except:
                    communication_problem = True
                    rospy.logwarn("Error in communications, trying again")
                    break
            # Both stop and 10 mm up movement to stop the robot
            self.trajectory_client.cancel_all_goals()
            #self.relative_move(0, 0, 0.001)
        return communication_problem

#
#  Test the different Robot_with_vaccum_gripper methods
#
# First, run this programs before running this one :
#
# roslaunch raiv_libraries ur3_bringup_cartesian.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml
# rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0
#
# Now, run :
# python robot_with_vaccum_gripper.py
#
if __name__ == '__main__':
    import rospy

    rospy.init_node('Robot_with_vaccum_gripper')
    myRobot = Robot_with_vaccum_gripper()
    input("============ Press `Enter` to test north and south moves ...")
    myRobot.go_to_initial_position()
    print('south')
    myRobot.small_move_to_south(0.1)
    print('north')
    myRobot.small_move_to_north(0.1)
    input("============ Press `Enter` to pick ...")
    pose_for_pick = geometry_msgs.Pose(
            geometry_msgs.Vector3(0.25, -0.13, 0.12), RobotUR.tool_down_pose
        )
    myRobot.pick(pose_for_pick)
    input("============ Press `Enter` to place ...")
    pose_for_place = geometry_msgs.Pose(
            geometry_msgs.Vector3(0.25, 0.13, 0.12), RobotUR.tool_down_pose
        )
    myRobot.place(pose_for_place)
    print("end")



