#!/usr/bin/env python3
import rclpy, csv, os, math
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class DistanceLogger(Node):
    def __init__(self):
        super().__init__('distance_logger')
        # --- params ---
        # self.declare_parameter('use_sim_time', True)
        self.declare_parameter('odom_topic', '/odometry')
        self.declare_parameter('csv_path', os.path.expanduser('~/.ros/distance_log.csv'))
        self.declare_parameter('log_every_n', 0)  # 0 = don't log each odom row
        self.declare_parameter('event_topic', '/event_mark')
        self.declare_parameter('events_csv_path', os.path.expanduser('~/.ros/event_log.csv'))
        self.declare_parameter('min_step_m', 0.001)
        self.declare_parameter('max_step_m', 5.0)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.csv_path   = os.path.expanduser(self.get_parameter('csv_path').value)
        self.log_every_n= int(self.get_parameter('log_every_n').value)
        self.event_topic= self.get_parameter('event_topic').value
        self.events_csv = os.path.expanduser(self.get_parameter('events_csv_path').value)
        self.min_step   = float(self.get_parameter('min_step_m').value)
        self.max_step   = float(self.get_parameter('max_step_m').value)

        # distance CSV (optional high-rate log)
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        self.csvf = open(self.csv_path, 'w', newline='', buffering=1)  # line-buffered
        self.w = csv.writer(self.csvf)
        self.w.writerow(['sim_time_sec','x','y','dx','dy','step_m','cumulative_m'])
        self.csvf.flush()

        # events CSV (sparse, labeled events only)
        os.makedirs(os.path.dirname(self.events_csv) or '.', exist_ok=True)
        self.ef = open(self.events_csv, 'w', newline='', buffering=1)
        self.ew = csv.writer(self.ef)
        self.ew.writerow(['sim_time_sec','label','x','y','cumulative_m'])
        self.ef.flush()

        # state
        self.prev_x = None
        self.prev_y = None
        self.last_x = None
        self.last_y = None
        self.cum = 0.0
        self._count = 0

        # subs
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos_profile_sensor_data)
        self.esub = self.create_subscription(String, self.event_topic, self.cb_event, 10)

        self.get_logger().info(f'Logging distance from {self.odom_topic} to {self.csv_path}')
        self.get_logger().info(f'Logging EVENTS from {self.event_topic} to {self.events_csv}')

    def cb_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.last_x, self.last_y = x, y  # for events
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            if self.log_every_n != 0:
                self.w.writerow([f'{t:.9f}', x, y, 0.0, 0.0, 0.0, self.cum]); self.csvf.flush()
            else:
                self.get_logger().info('First /odom received; distance accumulation started')
            return

        dx, dy = x - self.prev_x, y - self.prev_y
        step = math.hypot(dx, dy)

        if self.min_step <= step <= self.max_step:
            self.cum += step
            self.prev_x, self.prev_y = x, y
        elif step > self.max_step:
            self.prev_x, self.prev_y = x, y
            self.get_logger().warn(f'Pose jump {step:.2f} m; resetting anchor.')

        if self.log_every_n != 0:
            self._count += 1
            if self._count % self.log_every_n == 0:
                self.w.writerow([f'{t:.9f}', x, y, dx, dy, step, self.cum]); self.csvf.flush()

    def cb_event(self, msg: String):
        # stamp with ROS time so it aligns with /clock
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_x is None:
            self.get_logger().warn(f"Event '{msg.data}' received before any /odom; logging position as NaN")
            self.ew.writerow([f'{t:.9f}', msg.data, 'nan', 'nan', f'{self.cum:.6f}']); self.ef.flush()
            return
        self.ew.writerow([f'{t:.9f}', msg.data, f'{self.last_x:.6f}', f'{self.last_y:.6f}', f'{self.cum:.6f}'])
        self.ef.flush()
        self.get_logger().info(f"EVENT logged: {msg.data} @ dist={self.cum:.3f} m")

    def destroy_node(self):
        try:
            self.csvf.flush(); self.csvf.close()
            self.ef.flush(); self.ef.close()
            self.get_logger().info(f'Final distance: {self.cum:.3f} m')
        finally:
            super().destroy_node()

def main():
    rclpy.init()
    n = DistanceLogger()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()