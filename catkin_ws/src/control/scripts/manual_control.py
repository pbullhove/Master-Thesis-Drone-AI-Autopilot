#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

# Modified by Thomas Sundvoll
# Modified further by Peter Bull Hove

"""
Module for quadcopter control using keyboard commands.
Requires control/manual_control.py running.

-- Commands --
t: takeoff and start flight
l: land
L: automated landing
Arrows: linear flight
z: fly up
x: hover in the vertical axis.
c: fly downwards
v: rotate one way
b: rotate the other ways
1: take still photo bottom
2: take still photo front
d: start/stop data collection   (requres data collection module running)
p: toggle pid_on_off    (control/pid.py)

Subscribes to:
    None.
Publishes to:
    /cmd_vel: Twist - the commanded velocities from keyboard commands ot ardrone.
    /take_still_photo_front: Empty - capture photo from front camera and save to file
    /take_still_photo_bottom: Empty - capture photo from bottom camera and save to file
    /initiate_automated_landing: Empty - start automated landing from control/automated_landing.py
    /start_data_collection: Empty - start data collection from utilities/data_collection.py
    /pid_on_off: Bool  - Toggle pid controller on and off.
    /ardrone/takeoff: Empty - initate quadcopter takeoff.
    /ardrone/land: Empty - initiate quadcopter landing.
"""

import curses
import math

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Empty



class Velocity(object):

    def __init__(self, min_velocity, max_velocity, num_steps):
        assert min_velocity > 0 and max_velocity > 0 and num_steps > 0
        self._min = min_velocity
        self._max = max_velocity
        self._num_steps = num_steps
        if self._num_steps > 1:
            self._step_incr = (max_velocity - min_velocity) / (self._num_steps - 1)
        else:
            # If num_steps is one, we always use the minimum velocity.
            self._step_incr = 0

    def __call__(self, value, step):
        """
        Takes a value in the range [0, 1] and the step and returns the
        velocity (usually m/s or rad/s).
        """
        if step == 0:
            return 0

        assert step > 0 and step <= self._num_steps
        max_value = self._min + self._step_incr * (step - 1)
        return value * max_value

class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError, 'lineno out of bounds'
        height, width = self._screen.getmaxyx()
        y = (height / self._num_lines) * lineno
        x = 5
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()

class KeyTeleop():

    _interface = None

    _linear = None
    _angular = None

    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('key_vel', Twist)
        self._pub_ctr_switch = rospy.Publisher('/controller_switch', Bool)

        self._hz = rospy.get_param('~hz', 10)

        self._num_steps = rospy.get_param('~turbo/steps', 4)

        forward_min = rospy.get_param('~turbo/linear_forward_min', 0.5)
        forward_max = rospy.get_param('~turbo/linear_forward_max', 1.0)
        self._forward = Velocity(forward_min, forward_max, self._num_steps)

        backward_min = rospy.get_param('~turbo/linear_backward_min', 0.25)
        backward_max = rospy.get_param('~turbo/linear_backward_max', 0.5)
        self._backward = Velocity(backward_min, backward_max, self._num_steps)

        angular_min = rospy.get_param('~turbo/angular_min', 0.7)
        angular_max = rospy.get_param('~turbo/angular_max', 1.2)
        self._rotation = Velocity(angular_min, angular_max, self._num_steps)

    def run(self):
        self._linear = 0
        self._angular = 0

        rate = rospy.Rate(self._hz)
        while True:
            keycode = self._interface.read_key()
            if keycode:
                if self._key_pressed(keycode):
                    self._publish()
            else:
                self._publish()
                rate.sleep()

    def _get_twist(self, linear, angular):
        twist = Twist()
        if linear >= 0:
            twist.linear.x = self._forward(1.0, linear)
        else:
            twist.linear.x = self._backward(-1.0, -linear)
        twist.angular.z = self._rotation(math.copysign(1, angular), abs(angular))
        return twist

    def _key_pressed(self, keycode):
        movement_bindings = {
            curses.KEY_UP:    ( 1,  0),
            curses.KEY_DOWN:  (-1,  0),
            curses.KEY_LEFT:  ( 0,  1),
            curses.KEY_RIGHT: ( 0, -1),
        }
        speed_bindings = {
            ord(' '): (0, 0),
        }
        if keycode in movement_bindings:
            acc = movement_bindings[keycode]
            ok = False
            if acc[0]:
                linear = self._linear + acc[0]
                if abs(linear) <= self._num_steps:
                    self._linear = linear
                    ok = True
            if acc[1]:
                angular = self._angular + acc[1]
                if abs(angular) <= self._num_steps:
                    self._angular = angular
                    ok = True
            if not ok:
                self._interface.beep()
        elif keycode in speed_bindings:
            acc = speed_bindings[keycode]
            # Note: bounds aren't enforced here!
            if acc[0] is not None:
                self._linear = acc[0]
            if acc[1] is not None:
                self._angular = acc[1]

        elif keycode == ord('q'):
            rospy.signal_shutdown('Bye')
        else:
            return False

        return True

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Linear: %d, Angular: %d' % (self._linear, self._angular))
        self._interface.write_line(5, 'Use arrow keys to move, space to stop, q to exit.')
        self._interface.refresh()

        twist = self._get_twist(self._linear, self._angular)
        self._pub_cmd.publish(twist)


class SimpleKeyTeleop():
    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('cmd_vel', Twist)
        self._pub_ctr_switch = rospy.Publisher('/pid_on_off', Bool)
        self._pub_take_still_photo_front = rospy.Publisher('/take_still_photo_front', Empty)
        self._pub_take_still_photo_bottom = rospy.Publisher('/take_still_photo_bottom', Empty)
        self._pub_initiate_automated_landing = rospy.Publisher('/initiate_automated_landing', Empty)
        self._pub_start_data_collection = rospy.Publisher('/start_data_collection', Empty)
        self._pub_toggle_gt_feedback = rospy.Publisher('/toggle_gt_feedback', Empty)

        self._pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
        self._pub_land = rospy.Publisher('ardrone/land', Empty, queue_size=10)


        self._hz = rospy.get_param('~hz', 10)

        fw_rate = 0.1 # 0.8 default
        bw_rate = 0.1 # 0.5 default
        rot_rate = 0.5

        self._forward_rate = rospy.get_param('~forward_rate', fw_rate)
        self._backward_rate = rospy.get_param('~backward_rate', bw_rate)
        self._rotation_rate = rospy.get_param('~rotation_rate', rot_rate)
        self._last_pressed = {}
        self._angular = 0
        self._linear_x = 0
        self._linear_y = 0
        self._linear_z = 0
        self._take_off = 0
        self._land = 0
        self._controller_on = 0

    movement_bindings = {
        curses.KEY_UP:    ( 1,  0,  0,  1),
        curses.KEY_DOWN:  (-1,  0,  0,  1),
        curses.KEY_LEFT:  ( 0,  1,  0,  1),
        curses.KEY_RIGHT: ( 0, -1,  0,  1),
        ord('v'):         ( 0,  0,  1,  1),
        ord('b'):         ( 0,  0, -1,  1),
        ord('n'):         ( 0,  0,  0, 10),
    }

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self._set_velocity()
            self._publish()
            rate.sleep()

    def _get_twist(self, linear_x, linear_y, linear_z, angular):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.linear.z = linear_z
        twist.angular.z = angular
        return twist

    def _set_velocity(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.4:
                keys.append(a)
        linear_in_x = 0.0
        linear_in_y = 0.0
        angular = 0.0
        turbo = 1
        for k in keys:
            lx, ly, a, turbo = self.movement_bindings[k]
            linear_in_x += lx
            linear_in_y += ly
            angular += a
        if linear_in_x > 0:
            linear_in_x = linear_in_x * self._forward_rate *turbo
        else:
            linear_in_x = linear_in_x * self._backward_rate * turbo
        linear_in_y = linear_in_y * self._forward_rate * turbo
        angular = angular * self._rotation_rate
        self._angular = angular
        self._linear_x = linear_in_x
        self._linear_y = linear_in_y

    def _key_pressed(self, keycode):
        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode == ord('z'):
            self._linear_z = 0.4
        elif keycode == ord('x'):
            self._linear_z = 0
        elif keycode == ord('c'):
            self._linear_z = -0.4
        elif keycode == ord('t'): # take_off
            self._take_off = 1
        elif keycode == ord('l'): # land
            self._land = 1
        elif keycode == ord('p'):
            self._pub_ctr_switch.publish(Bool(1))
            self._controller_on = True
        elif keycode == ord('g'):
            self._pub_toggle_gt_feedback.publish(Empty())
        elif keycode == ord('o'):
            self._pub_ctr_switch.publish(Bool(0))
            self._controller_on = False
        elif keycode == ord('1'):
            self._pub_take_still_photo_bottom.publish(Empty())
        elif keycode == ord('2'):
            self._pub_take_still_photo_front.publish(Empty())
        elif keycode == ord('L'):
            self._pub_initiate_automated_landing.publish(Empty())
        elif keycode == ord('d'):
            self._pub_start_data_collection.publish(Empty())
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()

    def _publish(self):
        self._interface.clear()

        self._interface.write_line(2, 'Linear_x: %f' % (self._linear_x))
        self._interface.write_line(3, 'Linear_z: %f' % (self._linear_z))
        self._interface.write_line(4, 'Angular: %f' % (self._angular))
        self._interface.write_line(5, 'Take_off: %.1f, Land: %.1f' % (self._take_off, self._land))

        self._interface.write_line(6, 'Controller: %f' % (self._controller_on))


        self._interface.write_line(8, 'Use arrow keys to move, q to exit.')
        self._interface.refresh()

        twist = self._get_twist(self._linear_x, self._linear_y, self._linear_z, self._angular)
        self._pub_cmd.publish(twist)
        if self._take_off:
            self._pub_takeoff.publish(Empty())
            self._take_off = 0
        if self._land:
            self._pub_land.publish(Empty())
            self._land = 0

        # ctr_switch_msg = Bool()
        # ctr_switch_msg.data = self._controller_on
        #
        # self._pub_ctr_switch.publish(ctr_switch_msg)


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
