import copy
import rospy
import time
from std_msgs.msg import Bool
from environment import Environment
from robotUR import RobotUR

"""
Class used to establish connection with the robot and perform different actions such as move in all cardinal directions 
or pick and place an object. 
"""

class Robot_with_vaccum_gripper(RobotUR):
    def __init__(self, env, gripper_topic='switch_on_off', random_state_strategy='optimal'):
        super().__init__()
        self.gripper_topic = gripper_topic  # Gripper topic
        self.gripper_publisher = rospy.Publisher(self.gripper_topic, Bool, queue_size=10)  # Publisher for the gripper topic
        #self.image_controller = ImageController(image_topic='/usb_cam2/image_raw')
        self.environment_image = None
        self.random_state_strategy = random_state_strategy
        self.env = env

    def change_environment(self, new_env):
        self.env = new_env

    def calculate_relative_movement(self, relative_coordinates):
        absolute_coordinates_x = self.env.CARTESIAN_CENTER[0] - relative_coordinates[0]
        absolute_coordinates_y = self.env.CARTESIAN_CENTER[1] - relative_coordinates[1]
        current_pose = self.get_current_pose()
        x_movement = current_pose.pose.position.x - absolute_coordinates_x
        y_movement = current_pose.pose.position.y - absolute_coordinates_y
        return x_movement, y_movement

    def calculate_current_coordinates(self):
        """" Compute the TCP coordinates relative to the box axis (defined in Environment object)
        """
        absolute_coordinate_x = self.get_current_pose().pose.position.x
        absolute_coordinate_y = self.get_current_pose().pose.position.y
        relative_coordinate_x = self.env.CARTESIAN_CENTER[0] - absolute_coordinate_x
        relative_coordinate_y = self.env.CARTESIAN_CENTER[1] - absolute_coordinate_y
        return [relative_coordinate_x, relative_coordinate_y]

    # Action north: positive x
    def take_north(self, distance=None):
        if distance == None:
            distance = self.env.ACTION_DISTANCE
        self.relative_move(distance, 0, 0)

    # Action south: negative x
    def take_south(self, distance=None):
        if distance == None:
            distance = self.env.ACTION_DISTANCE
        self.relative_move(-distance, 0, 0)

    # Action east: negative y
    def take_east(self, distance=None):
        if distance == None:
            distance = self.env.ACTION_DISTANCE
        self.relative_move(0, -distance, 0)

    # Action west: positive y
    def take_west(self, distance=None):
        if distance == None:
            distance = self.env.ACTION_DISTANCE
        self.relative_move(0, distance, 0)

    def take_random_state(self):
        # Move robot to random positions using relative moves. Get coordinates
        relative_coordinates = Environment(self.env).generate_random_state(self.environment_image, self.random_state_strategy)
        # Calculate the new coordinates
        x_movement, y_movement = self.calculate_relative_movement(relative_coordinates)
        # Move the robot to the random state
        self.relative_move(x_movement, y_movement, 0)

    def take_random_state_fall(self):
        # Move robot to random positions using relative moves. Get coordinates
        relative_coordinates = Environment(self.env).generate_random_state_fall(self.environment_image, self.random_state_strategy)
        # Calculate the new coordinates
        x_movement, y_movement = self.calculate_relative_movement(relative_coordinates)
        # Move the robot to the random state
        self.relative_move(x_movement, y_movement, 0)

    def send_gripper_message(self, msg, timer=2, n_msg=10):
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

    # Action pick: Pick and place
    def take_pick(self, no_rotation = False):
        # The robot goes down until the gripper touch something (table or object)
        # Detection provided by /contact topic
        # If no_rotation = True, it means that the gripper must release the object and mustn't turn.
        # To be used in recording images for NN training

        def change_plan_speed(plan, new_speed):
            """
            Function used for changing Robot velocity of a cartesian path once the movement have been planned.
            :param plan: RobotTrajectory object. For example, the one calculated by compute_cartesian_path() MoveGroup function.
            :param new_speed: speed factor of the robot, been 1 the original speed and 0 the minimum.
            :return: RobotTrajectory object (new plan).
            """
            new_plan = plan
            n_joints = len(plan.joint_trajectory.joint_names)
            n_points = len(plan.joint_trajectory.points)

            points = []
            for i in range(n_points):
                plan.joint_trajectory.points[i].time_from_start = plan.joint_trajectory.points[
                                                                      i].time_from_start / new_speed
                velocities = []
                accelerations = []
                positions = []
                for j in range(n_joints):
                    velocities.append(plan.joint_trajectory.points[i].velocities[j] * new_speed)
                    accelerations.append(plan.joint_trajectory.points[i].accelerations[j] * new_speed)
                    positions.append(plan.joint_trajectory.points[i].positions[j])
                point = plan.joint_trajectory.points[i]
                point.velocities = velocities
                point.accelerations = accelerations
                point.positions = positions
                points.append(point)
            new_plan.joint_trajectory.points = points
            return new_plan

        def back_to_original_pose():
            """
            Function used to go back to the original height once a vertical movement has been performed.
            :param robot: robot_controller.robot.py object
            :return:
            """
            distance = self.env.CARTESIAN_CENTER[2] - self.get_current_pose().pose.position.z
            self.relative_move(0, 0, distance)

        def down_movement(movement_speed):
            """
            This function performs the down movement of the pick action.

            It creates an asynchronous move group trajectory planning. This way the function is able to receive distance
            messages while the robot is moving and stop it once the robot is in contact with an object.

            Finally, when there is any problems with the communications the movement is stopped and
            communication_problem boolean flag is set to True. It is considered that there is a problem with
            communications when the robot is not receiving any distance messages during 200 milli-seconds (timeout=0.2)

            :param robot: robot_controller.robot.py object
            :return: communication_problem flag
            """

            contact_ok = rospy.wait_for_message('contact', Bool).data  # We retrieve sensor contact
            communication_problem = False

            if not contact_ok:  # If the robot is already in contact with an object, no movement is performed
                waypoints = []
                wpose = self.get_current_pose().pose
                wpose.position.z -= (wpose.position.z)  # Third move sideways (z)
                waypoints.append(copy.deepcopy(wpose))

                (plan, fraction) = self.move_group.compute_cartesian_path(
                    waypoints,  # waypoints to follow
                    0.01,  # eef_step
                    0.0)  # jump_threshold

                plan = change_plan_speed(plan, movement_speed)
                self.move_group.execute(plan, wait=False)

                while not contact_ok:
                    try:
                        contact_ok = rospy.wait_for_message('contact', Bool, 0.2).data  # We retrieve sensor contact
                    except:
                        communication_problem = True
                        rospy.logwarn("Error in communications, trying again")
                        break

                # Both stop and 10 mm up movement to stop the robot
                self.move_group.stop()
                self.relative_move(0, 0, 0.001)

            return communication_problem

        communication_problem = True
        while communication_problem:  # Infinite loop until the movement is completed
            communication_problem = down_movement(movement_speed=0.15)

        self.send_gripper_message(True, timer=1)  # We turn on the gripper

        back_to_original_pose()  # Back to the original pose

        object_gripped = rospy.wait_for_message('object_gripped', Bool).data
        '''
        if object_gripped and no_rotation == False:  # If we have gripped an object we place it into the desired point
            self.take_place()
        
        elif object_gripped and no_rotation == True:
            print("objet attrapé")
            #robot2.go_to_initial_pose()
            self.take_random_state()

            self.send_gripper_message(False)  # We turn off the gripper

        else:
            print("objet attrapé")
            #robot2.go_to_initial_pose()
            #robot2.take_random_state_fall()

            self.send_gripper_message(False)  # We turn off the gripper
        '''

        return object_gripped

    # Function to define the place for placing the grasped objects
    def take_place(self):
        # First, we get the cartesian coordinates of one of the corner
        x_box, y_box = Environment(self.env).get_relative_corner('se')
        x_move, y_move = self.calculate_relative_movement([x_box, y_box])
        # We move the robot to the corner of the box
        self.relative_move(x_move, y_move, 0)
        # We calculate the trajectory for our robot to reach the box
        trajectory_x = self.get_current_pose().pose.position.x - self.env.PLACE_CARTESIAN_CENTER[0]
        trajectory_y = self.get_current_pose().pose.position.y - self.env.PLACE_CARTESIAN_CENTER[1]
        trajectory_z = - self.env.CARTESIAN_CENTER[2] + self.env.PLACE_CARTESIAN_CENTER[2]
        # We move the robot to the coordinates desired to place the object
        self.relative_move(0, 0, trajectory_z)
        self.relative_move(0, trajectory_y, 0)
        self.relative_move(trajectory_x, 0, 0)
        # Then, we left the object
        self.relative_move(0, 0, -0.05)
        # Then, we switch off the vacuum gripper so the object can be placed
        self.send_gripper_message(False)
        # Wait some seconds, in order to the msg to arrive to the gripper
        time.sleep(2)
        # Then the robot goes up
        self.relative_move(0, 0, 0.05)
        # get environment image
        # self.environment_image, w, l = self.image_controller.get_image()
        # self.image_controller.record_image(self.environment_image, True)
        # Final we put the robot in the center of the box, the episode should finish now
        self.go_to_joint_state(self.env.ANGULAR_CENTER)

    def go_to_initial_pose(self):
        """
        Go to the center of the box defined by joint coordinates
        """
        target_reached = self.go_to_joint_state(self.env.ANGULAR_CENTER)
        if target_reached:
            print("Target reachead")
        else:
            print("Target not reached")

    def go_up_before_changing_box(self):
        up_point_pose = self.get_current_pose().pose
        up_point_pose.position.z = 0.45
        target_up = self.go_to_pose_goal(up_point_pose)
        if target_up:
            print("Target reachead")
        else:
            print("Target not reached")


#
#  Test the different Robot_with_vaccum_gripper methods
#
if __name__ == '__main__':
    from environment import Env_cam_bas
    import rospy

    rospy.init_node('Robot_with_vaccum_gripper')
    env = Env_cam_bas()
    myRobot = Robot_with_vaccum_gripper(env)
    myRobot.take_north()
    myRobot.take_south()
    # Try to take an object
    myRobot.take_pick()
    # Release the object
    myRobot.take_place()
    # display X,Y coord. relative to the box axis defined by the environment (here a box defined by Env_cam_bas)
    myRobot.calculate_current_coordinates()
    myRobot.take_random_state()
    # Go to the center of the box
    myRobot.go_to_initial_pose()
    print("Coordinates of TCP in box axis")
    print(myRobot.calculate_current_coordinates())
    # la descente se fait, lors d'un contact, arrêt du robot puis déclnchement de la pompe mais pas de remonté, la méthode s'arrête et affiche False.


