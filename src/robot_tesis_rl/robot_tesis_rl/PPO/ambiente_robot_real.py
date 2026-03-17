#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────
#  ENTORNO GYM + ROS 2 PARA ROBOT REAL (sin Gazebo)
#  -----------------------------------------------
#  • Eliminado todo lo relacionado con Gazebo y obstáculos
#  • Mantiene la función de recompensa y lógica de RL
#  • Usa odometría y LiDAR del robot real
#  • Visualización de meta en RViz2 mediante Marker
#  • Posición inicial del robot es donde esté físicamente
# ──────────────────────────────────────────────────────────

import threading
import time
import math
import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

import gymnasium as gym


device_random = random


def seed_all(seed: int):
    """
    Fija la semilla para random y numpy para reproducibilidad.
    """
    device_random.seed(seed)
    np.random.seed(seed)


class RobotReal(gym.Env, Node):
    """
    Entorno RL para robot diferencial real con ROS2.
    Sin simulación Gazebo, trabaja directamente con el robot físico.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        step_duration: float = 0.10,  # Más lento para robot real
        goal_area: tuple = (-3.0, 3.0, -3.0, 3.0),  # Área más pequeña para robot real
    ):
        # ─── Iniciar nodo y executor ROS2 (multithreaded) ───
        Node.__init__(self, "rl_robot_real_env")
        self.step_dt = step_duration

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        threading.Thread(target=self._executor.spin, daemon=True).start()

        # ─── Parámetros del entorno real ───
        self.goal_area = goal_area

        # ─── Parámetros de recompensa ───
        self.rg = 200    # recompensa al llegar al objetivo
        self.rc = -150   # penalización por colisión
        self.rep = -100  # penalización por no llegar a tiempo
        self.goal_thresh = 1.0       # umbral más pequeño para robot real
        self.collision_thresh = 0.6  # umbral más conservador para robot real
        self.safety_margin = 1.0

        # ─── Variables de episodios ───
        self.paso_en_episodio = 0
        self.max_steps = 60000  # Menos pasos para robot real

        # ─── Truncamiento adaptativo ───
        self.no_prog_limit = 150000
        self.min_prog_thresh = 0.03
        self.no_prog_counter = 0

        self.verbose = False

        # ─── Acción ───
        self.v_max = 0.4 # 0.4              # Velocidad más conservadora para robot real
        self.w_max = 0.5 # 0.5              # Velocidad angular más conservadora
        self.action_space = gym.spaces.Discrete(3)

        # Tabla de acciones más conservadora
        self._action_table = np.array([
            [ 1.0,  0.0],   # FWD
            [ 0.0, +0.6],   # LEFT
            [ 0.0, -0.6],   # RIGHT
        ], dtype=np.float32)

        # ─── Observación: LiDAR 360° normalizado + extras ───
        self.n_sect = 36
        self.r_min, self.r_max = 0.15, 6.0  # Rango más conservador

        low = np.concatenate([
            np.zeros(self.n_sect),
            [0.0, -1.0, -1.0, -1.0, -1.0]
        ]).astype(np.float32)
        high = np.concatenate([
            np.ones(self.n_sect),
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # ─── Publicadores y subscriptores ROS ───
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)

        # ─── Variables de estado del robot ───
        self.robot_ready = False
        self.initial_pose_set = False
        self._reset_internal_vars()
        
        # Esperar a que lleguen datos del robot
        self.get_logger().info("🤖 Esperando datos del robot real...")
        self._wait_for_robot_data()
        
        self.get_logger().info("🚀 Entorno para robot real inicializado.")

    def _wait_for_robot_data(self, timeout=10.0):
        """Espera a que lleguen datos del robot antes de continuar."""
        start_time = time.time()
        while not self.robot_ready and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            # Verificar que tenemos datos de sensores
            if hasattr(self, 'pos_x') and hasattr(self, 'lidar'):
                self.robot_ready = True
                self.get_logger().info("✅ Robot conectado y sensores activos")
                break
        
        if not self.robot_ready:
            raise RuntimeError("❌ No se pudieron obtener datos del robot en el tiempo esperado")

    def _reset_internal_vars(self):
        """Variables de estado al inicio de cada episodio."""
        self.pos_x = self.pos_y = self.yaw = 0.0
        self.v_act = self.w_act = 0.0
        self.lidar = np.ones(self.n_sect, dtype=np.float32) * self.r_max
        self.target_x = self.target_y = 0.0
        self.paso_en_episodio = 0
        self.no_prog_counter = 0

    # ─────────────────────────────────────────────
    # CALLBACKS ROS
    # ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Callback de odometría del robot real."""
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        _, _, self.yaw = self._quat_to_rpy(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.v_act = msg.twist.twist.linear.x
        self.w_act = msg.twist.twist.angular.z

    def _scan_cb(self, msg: LaserScan):
        """Callback de LiDAR del robot real."""
        rays_sec = max(1, len(msg.ranges) // self.n_sect)
        full_vals = []
        for i in range(self.n_sect):
            sect = msg.ranges[i * rays_sec : (i + 1) * rays_sec]
            m = min(
                (v if not math.isinf(v) and not math.isnan(v) else self.r_max)
                for v in sect
            )
            full_vals.append(min(m, self.r_max))
        self.lidar = np.array(full_vals, dtype=np.float32)

    # ─────────────────────────────────────────────
    # UTILIDADES GEOMÉTRICAS
    # ─────────────────────────────────────────────

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        """Convierte cuaternión a ángulos roll-pitch-yaw."""
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        return 0.0, 0.0, math.atan2(t3, t4)

    @staticmethod
    def _yaw_to_quat(yaw):
        """Convierte ángulo yaw a cuaternión."""
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        return q

    def _dist_to_goal(self):
        """Distancia Euclidiana al objetivo."""
        dx = self.target_x - self.pos_x
        dy = self.target_y - self.pos_y
        return math.hypot(dx, dy)

    def _yaw_error(self):
        """Error angular hacia el objetivo."""
        angle_to_goal = math.atan2(self.target_y - self.pos_y,
                                  self.target_x - self.pos_x)
        err = angle_to_goal - self.yaw
        return (err + math.pi) % (2 * math.pi) - math.pi

    def _publish_goal_marker(self):
        """Publica marcador visual del objetivo en RViz."""
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.target_x)
        m.pose.position.y = float(self.target_y)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.3
        m.scale.y = 0.3
        m.scale.z = 0.3
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        self.marker_pub.publish(m)

    # ─────────────────────────────────────────────
    # RESET (episodio)
    # ─────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        Reinicia episodio para el robot real.
        La posición inicial es donde esté el robot físicamente.
        """
        super().reset(seed=seed)
        if seed is not None:
            seed_all(seed)

        # Detener el robot
        self._publish_cmd(0.0, 0.0)
        time.sleep(0.5)  # Pausa para asegurar que se detenga

        # Reiniciar variables
        self._reset_internal_vars()
        self.episode_start_time = time.time()

        # Esperar datos actualizados del robot
        rclpy.spin_once(self, timeout_sec=0.1)

        # La posición inicial es donde esté el robot ahora
        initial_x = self.pos_x
        initial_y = self.pos_y
        
        # Generar objetivo aleatorio en área permitida
        # Asegurar distancia mínima del objetivo
        min_goal_dist = 1.5
        max_attempts = 50
        
        for _ in range(max_attempts):
            self.target_x = random.uniform(*self.goal_area[:2])
            self.target_y = random.uniform(*self.goal_area[2:])
            dist = math.hypot(self.target_x - initial_x, self.target_y - initial_y)
            if dist > min_goal_dist:
                break
        else:
            # Si no se encontró posición válida, usar una por defecto
            self.target_x = initial_x + 2.0
            self.target_y = initial_y + 0.0

        # Publicar objetivo
        self._publish_goal_pose()
        self._publish_goal_marker()
        self.prev_dist = self._dist_to_goal()

        self.get_logger().info(f"🎯 Nuevo episodio iniciado")
        self.get_logger().info(f"   Robot en: ({initial_x:.2f}, {initial_y:.2f})")
        self.get_logger().info(f"   Objetivo: ({self.target_x:.2f}, {self.target_y:.2f})")
        self.get_logger().info(f"   Distancia: {self.prev_dist:.2f}m")

        return self._get_obs(), {}

    # ─────────────────────────────────────────────
    # STEP (acción)
    # ─────────────────────────────────────────────

    def step(self, action):
        """Ejecuta paso en el robot real."""
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1️⃣ CONVERSIÓN DE ACCIÓN A VELOCIDADES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        assert self.action_space.contains(action), "Acción fuera de rango"
        v_norm, w_norm = self._action_table[action]
        v = v_norm * self.v_max
        w = w_norm * self.w_max
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 2️⃣ EJECUCIÓN EN ROBOT REAL
        # ═══════════════════════════════════════════════════════════════════════════════
        
        self._publish_cmd(v, w)
        time.sleep(self.step_dt)  # Esperar que se ejecute la acción
        
        # Obtener datos actualizados
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 3️⃣ LECTURA DE SENSORES
        # ═══════════════════════════════════════════════════════════════════════════════
        
        error_pos = self._dist_to_goal()
        error_angular = self._yaw_error()
        min_lidar = float(min(self.lidar))
        max_lidar = float(max(self.lidar))
        v_cur = self.v_act
        w_cur = self.w_act
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 4️⃣ CÁLCULO DE RECOMPENSA (misma lógica que simulación)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        raw_prog = self.prev_dist - error_pos
        prog = max(0.0, min(1.0, raw_prog / (self.prev_dist + 1e-6)))
        
        smooth_drive = max(0.0, v_cur) * max(0.0, math.cos(error_angular))
        spin_pen = -1.5 * abs(w_cur)
        stall_pen = -0.8 if abs(v) < 0.03 else 0.0
        obs_pen = -5.0 * max(0.0, (self.collision_thresh - min_lidar) / self.collision_thresh)
        safe_bonus = 2.0 if min_lidar > self.safety_margin else 0.0
        alignment_bonus = 3.0 if abs(error_angular) < 0.1 else 0.0
        
        reward = (
            8.0 * prog +
            3.0 * smooth_drive +
            spin_pen +
            stall_pen +
            obs_pen +
            safe_bonus +
            alignment_bonus
        )
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 5️⃣ CONDICIONES DE TÉRMINO
        # ═══════════════════════════════════════════════════════════════════════════════
        
        done = False
        estado = "🚗 En ruta"
        
        if error_pos < self.goal_thresh:
            reward = self.rg
            done = True
            estado = "🏆 Objetivo alcanzado"
        elif min_lidar < self.collision_thresh:
            reward = self.rc
            done = True
            estado = "💥 Colisión detectada"
        
        self.prev_dist = error_pos
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 6️⃣ TRUNCAMIENTO
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if prog < self.min_prog_thresh:
            self.no_prog_counter += 1
        else:
            self.no_prog_counter = 0
        
        self.paso_en_episodio += 1
        
        time_out = self.paso_en_episodio >= self.max_steps
        stuck_too_long = self.no_prog_counter >= self.no_prog_limit
        truncated = time_out or stuck_too_long
        
        if truncated:
            reward = self.rep
            estado = "⌛ Timeout" if time_out else "🛑 Sin progreso"
            # Detener robot si se trunca
            self._publish_cmd(0.0, 0.0)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 7️⃣ OBSERVACIÓN
        # ═══════════════════════════════════════════════════════════════════════════════
        
        obs = self._get_obs()
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 8️⃣ LOGGING
        # ═══════════════════════════════════════════════════════════════════════════════
        
        should_log = (
            self.paso_en_episodio % 20 == 0 or  # Menos frecuente para robot real
            done or truncated or
            min_lidar < self.collision_thresh * 1.5
        )
        
        if should_log:
            log = self.get_logger().info
            log(f"Step {self.paso_en_episodio:3d} | {estado}")
            log(f"  🎯 Dist: {error_pos:.2f}m  🧭 Yaw: {math.degrees(error_angular):+.0f}°")
            log(f"  🚗 Vel: v={v:.2f} w={w:.2f}  🔦 LiDAR: {min_lidar:.2f}m")
            log(f"  💰 Reward: {reward:+.2f}")
            
            if done or truncated:
                log("─" * 50)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 9️⃣ INFO
        # ═══════════════════════════════════════════════════════════════════════════════
        
        info = {
            "distance": error_pos,
            "error_yaw": error_angular,
            "min_lidar": min_lidar,
            "max_lidar": max_lidar,
            "v_cmd": v,
            "w_cmd": w,
            "progress": prog,
            "reward_components": {
                "progress": 8.0 * prog,
                "smooth_drive": 3.0 * smooth_drive,
                "spin_penalty": spin_pen,
                "stall_penalty": stall_pen,
                "obstacle_penalty": obs_pen,
                "safety_bonus": safe_bonus,
                "alignment_bonus": alignment_bonus
            },
            "truncated_reason": (
                "timeout" if time_out else
                "no_progress" if stuck_too_long else
                None
            )
        }
        
        return obs, reward, done, truncated, info

    def _get_obs(self):
        """Construye observación normalizada."""
        lidar_n = np.clip(
            (self.lidar - self.r_min) / (self.r_max - self.r_min),
            0.0, 1.0
        )
        d_n = np.clip(self._dist_to_goal() / 6.0, 0.0, 1.0)  # Escala más pequeña
        dir_x = math.cos(self._yaw_error())
        dir_y = math.sin(self._yaw_error())
        v_n = max(0.0, self.v_act / self.v_max)
        w_n = self.w_act / self.w_max

        obs = np.concatenate([
            lidar_n,
            [d_n, dir_x, dir_y, v_n, w_n]
        ]).astype(np.float32)

        return obs

    def render(self, mode="human"):
        """Render simple para robot real."""
        if mode == "rgb_array":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(self.pos_x, self.pos_y, 'bo', markersize=10, label="Robot")
            ax.plot(self.target_x, self.target_y, 'rx', markersize=15, label="Goal")
            
            # Dibujar orientación del robot
            dx = 0.3 * math.cos(self.yaw)
            dy = 0.3 * math.sin(self.yaw)
            ax.arrow(self.pos_x, self.pos_y, dx, dy, head_width=0.1, color='blue')
            
            ax.set(xlim=(-4, 4), ylim=(-4, 4), title="Robot Real - Navegación RL")
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal')
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            h, w = fig.canvas.get_width_height()
            plt.close(fig)
            return img.reshape((h, w, 3))

    def close(self):
        """Cierra el entorno y detiene el robot."""
        self.get_logger().info("🛑 Deteniendo robot y cerrando entorno...")
        self._publish_cmd(0.0, 0.0)
        time.sleep(1.0)  # Asegurar que se detenga
        self._executor.shutdown()

    # ─────────────────────────────────────────────
    # UTILIDADES ROS
    # ─────────────────────────────────────────────

    def _publish_cmd(self, v, w):
        """Publica comando de velocidad."""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _publish_goal_pose(self):
        """Publica posición del objetivo."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = float(self.target_x)
        msg.pose.position.y = float(self.target_y)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)


# ──────────────────────────────────────────────────────────
# TEST RÁPIDO PARA ROBOT REAL
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rclpy.init()
    try:
        env = RobotReal()
        obs, _ = env.reset()
        print(f"✅ Entorno iniciado. Obs shape: {obs.shape}")
        print("🤖 Robot listo para pruebas...")
        
        # Prueba simple de unos pasos
        for i in range(10):
            action = env.action_space.sample()
            obs, rew, done, truncated, info = env.step(action)
            print(f"Paso {i+1}: reward={rew:.2f}, dist={info['distance']:.2f}m")
            if done or truncated:
                print("Episodio terminado")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'env' in locals():
            env.close()
        rclpy.shutdown()
