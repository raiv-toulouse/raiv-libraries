from robotUR import RobotUR
import time
import socket
from std_srvs.srv import Trigger, TriggerRequest


class Robot_with_gripper(RobotUR):
    def __init__(self):
        super().__init__()

    def open_gripper(self):
        """

        @return:
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.tcp_robot_ip, self.tcp_port))
        tcp_command = "set_digital_out(8,False)\n"
        tcp_socket.send(tcp_command)
        tcp_socket.close()
        time.sleep(0.5)
        play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
        play = TriggerRequest()
        play_service(play)
        time.sleep(0.5)

    def close_gripper(self):
        """

        @return:
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.tcp_robot_ip, self.tcp_port))
        tcp_command = "set_digital_out(8,True)\n"
        tcp_socket.send(tcp_command)
        tcp_socket.close()
        time.sleep(0.5)
        play_service = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
        play = TriggerRequest()
        play_service(play)
        time.sleep(0.5)

#
#  Test the different RobotUR methods
#
if __name__ == '__main__':
    import rospy
    myRobot = Robot_with_gripper()
    rospy.init_node('robot_with_gripper')
    print("Press ENTER to continue")
    input()
    myRobot.open_gripper()
    print("Press ENTER to continue")
    input()
    myRobot.close_gripper()
