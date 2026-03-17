#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# test_robot_real.py - Testing para robot real (sin Gazebo)
# ────────────────────────────────────────────────────────────────
import os
import csv
import time
import numpy as np
import gymnasium as gym
from datetime import datetime

import rclpy
from stable_baselines3 import PPO
import robot_env  # El ambiente para robot real

class RobotRealTester:
    def __init__(self, model_path, manual_goals=False):
        """
        Inicializa el tester para robot real
        
        Args:
            model_path: Ruta al archivo .zip del modelo PPO entrenado
            manual_goals: Si True, permite ingresar objetivos manualmente
        """
        # Inicializar ROS2
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass
        
        # Crear ambiente para robot real
        print("🤖 Conectando con robot real...")
        self.env_wrapper = gym.make("RobotReal-v0", step_duration=0.15)
        self.env = self.env_wrapper.unwrapped
        
        self.manual_goals = manual_goals
        print(f"🎯 Modo: {'Manual' if manual_goals else 'Automático'}")
        
        # Cargar modelo entrenado
        print(f"🧠 Cargando modelo desde: {model_path}")
        self.model = PPO.load(model_path)
        print("✅ Modelo cargado exitosamente")
        
        # Configurar directorio de resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"test_real_robot_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Resultados se guardarán en: {self.results_dir}")
        
        self.goal_tolerance = 1.0  # Más estricto para robot real
        
        # Esperar a que el robot esté listo
        print("⏳ Esperando que el robot esté listo...")
        time.sleep(2.0)
        
    def test_single_goal(self, goal_x=None, goal_y=None, max_steps=4000000):
        """
        Prueba navegación a un objetivo único en robot real
        """
        if goal_x is None or goal_y is None:
            if self.manual_goals:
                print("📍 Ingresa las coordenadas del objetivo:")
                goal_x = float(input("  X: "))
                goal_y = float(input("  Y: "))
            else:
                # Objetivo aleatorio cerca del robot
                goal_x = self.env.pos_x + np.random.uniform(-2.0, 2.0)
                goal_y = self.env.pos_y + np.random.uniform(-2.0, 2.0)
        
        print(f"🎯 Navegando hacia objetivo: ({goal_x:.2f}, {goal_y:.2f})")
        
        csv_filename = os.path.join(
            self.results_dir, 
            f"single_goal_{goal_x:.1f}_{goal_y:.1f}.csv"
        )
        
        # 🆕 NUEVO: Archivo CSV para velocidades
        velocity_csv_filename = os.path.join(
            self.results_dir, 
            f"velocities_{goal_x:.1f}_{goal_y:.1f}.csv"
        )
        
        # Resetear ambiente (sin cambiar posición física del robot)
        obs, _ = self.env_wrapper.reset()
        
        # Configurar objetivo manualmente
        self.env.target_x = goal_x
        self.env.target_y = goal_y
        self.env._publish_goal_pose()
        self.env._publish_goal_marker()
        
        print(f"📍 Posición inicial del robot: ({self.env.pos_x:.2f}, {self.env.pos_y:.2f})")
        print(f"🎯 Objetivo establecido en: ({goal_x:.2f}, {goal_y:.2f})")
        
        initial_distance = np.sqrt((goal_x - self.env.pos_x)**2 + (goal_y - self.env.pos_y)**2)
        print(f"📏 Distancia inicial: {initial_distance:.2f}m")
        
        # Confirmar inicio
        if self.manual_goals:
            input("Presiona Enter para iniciar la navegación...")
        
        trajectory = []
        velocity_data = []  # 🆕 NUEVO: Lista para datos de velocidad
        step_count = 0
        done = False
        
        print("🚀 Iniciando navegación autónoma...")
        start_time = time.time()
        
        while not done and step_count < max_steps:
            # Obtener acción del modelo
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Ejecutar acción en robot real
            obs, reward, terminated, truncated, info = self.env_wrapper.step(action)
            done = terminated or truncated
            
            # Obtener posición actual del robot
            x, y = self.env.pos_x, self.env.pos_y
            
            # Guardar datos de trayectoria
            trajectory.append({
                'step': step_count,
                'x': x,
                'y': y,
                'goal_x': goal_x,
                'goal_y': goal_y,
                'reward': reward,
                'time': time.time() - start_time,
                'robot_yaw': self.env.yaw,
                'distance_to_goal': info.get('distance', np.sqrt((x - goal_x)**2 + (y - goal_y)**2)),
                'min_lidar': info.get('min_lidar', 0.0),
                'v_cmd': info.get('v_cmd', 0.0),
                'w_cmd': info.get('w_cmd', 0.0),
                'robot_v': self.env.v_act,
                'robot_w': self.env.w_act
            })
            
            # 🆕 NUEVO: Guardar datos de velocidad por separado
            velocity_data.append({
                'step': step_count,
                'time': time.time() - start_time,
                'v_linear': self.env.v_act,     # Velocidad lineal real del robot
                'w_angular': self.env.w_act,    # Velocidad angular real del robot
                'v_cmd': info.get('v_cmd', 0.0),    # Comando de velocidad lineal enviado
                'w_cmd': info.get('w_cmd', 0.0)     # Comando de velocidad angular enviado
            })
            
            step_count += 1
            
            # Mostrar progreso cada 20 pasos (menos frecuente para robot real)
            if step_count % 20 == 0:
                distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
                min_obs = info.get('min_lidar', 0.0)
                print(f"Paso {step_count}: Pos=({x:.2f}, {y:.2f}), Dist={distance:.2f}m, LiDAR_min={min_obs:.2f}m")
        
        # Detener robot al finalizar
        self.env._publish_cmd(0.0, 0.0)
        time.sleep(0.5)
        
        # Guardar trayectoria en CSV
        self._save_trajectory_csv(trajectory, csv_filename)
        
        # 🆕 NUEVO: Guardar velocidades en CSV separado
        self._save_velocity_csv(velocity_data, velocity_csv_filename)
        
        # Verificar éxito
        final_distance = np.sqrt((trajectory[-1]['x'] - goal_x)**2 + 
                                (trajectory[-1]['y'] - goal_y)**2)
        success = final_distance < self.goal_tolerance
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 Resultado: {'✅ ÉXITO' if success else '❌ FALLO'}")
        print(f"   Distancia final: {final_distance:.2f}m")
        print(f"   Pasos totales: {step_count}")
        print(f"   Tiempo: {elapsed_time:.1f}s")
        print(f"   Velocidad promedio: {initial_distance/elapsed_time:.2f}m/s")
        print(f"   CSV trayectoria: {csv_filename}")
        print(f"   CSV velocidades: {velocity_csv_filename}")  # 🆕 NUEVO
        
        return success
    
    def test_waypoint_route(self, waypoints=None, max_steps_per_goal=300):
        """
        Prueba navegación por una ruta de waypoints en robot real
        """
        if waypoints is None:
            waypoints = self._get_waypoints_input()
        
        print(f"🗺️ Navegando ruta de {len(waypoints)} waypoints en robot real")
        
        csv_filename = os.path.join(
            self.results_dir, 
            f"waypoint_route_{len(waypoints)}_points.csv"
        )
        
        # 🆕 NUEVO: Archivo CSV para velocidades en rutas
        velocity_csv_filename = os.path.join(
            self.results_dir, 
            f"velocities_route_{len(waypoints)}_points.csv"
        )
        
        full_trajectory = []
        velocity_data = []  # 🆕 NUEVO
        route_stats = {
            'waypoints_reached': 0,
            'total_steps': 0,
            'total_time': 0,
            'success_rate': 0
        }
        
        start_time = time.time()
        global_step = 0
        
        for i, (goal_x, goal_y) in enumerate(waypoints):
            print(f"\n🎯 Waypoint {i+1}/{len(waypoints)}: ({goal_x:.2f}, {goal_y:.2f})")
            
            # Solo resetear en el primer waypoint
            if i == 0:
                obs, _ = self.env_wrapper.reset()
                print(f"   📍 Posición inicial: ({self.env.pos_x:.2f}, {self.env.pos_y:.2f})")
            else:
                print(f"   📍 Posición actual: ({self.env.pos_x:.2f}, {self.env.pos_y:.2f})")
            
            # Configurar nuevo objetivo
            self.env.target_x = goal_x
            self.env.target_y = goal_y
            self.env._publish_goal_pose()
            self.env._publish_goal_marker()
            
            if self.manual_goals:
                input(f"   ▶️ Presiona Enter para ir al waypoint {i+1}...")
            
            step_count = 0
            done = False
            
            while not done and step_count < max_steps_per_goal:
                # Obtener acción del modelo
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Ejecutar acción
                obs, reward, terminated, truncated, info = self.env_wrapper.step(action)
                done = terminated or truncated
                
                # Obtener posición actual del robot
                x, y = self.env.pos_x, self.env.pos_y
                
                # Guardar datos
                full_trajectory.append({
                    'global_step': global_step,
                    'waypoint': i + 1,
                    'local_step': step_count,
                    'x': x,
                    'y': y,
                    'goal_x': goal_x,
                    'goal_y': goal_y,
                    'reward': reward,
                    'time': time.time() - start_time,
                    'robot_yaw': self.env.yaw,
                    'distance_to_goal': info.get('distance', np.sqrt((x - goal_x)**2 + (y - goal_y)**2)),
                    'min_lidar': info.get('min_lidar', 0.0),
                    'v_cmd': info.get('v_cmd', 0.0),
                    'w_cmd': info.get('w_cmd', 0.0),
                    'robot_v': self.env.v_act,
                    'robot_w': self.env.w_act
                })
                
                # 🆕 NUEVO: Guardar datos de velocidad
                velocity_data.append({
                    'global_step': global_step,
                    'waypoint': i + 1,
                    'local_step': step_count,
                    'time': time.time() - start_time,
                    'v_linear': self.env.v_act,
                    'w_angular': self.env.w_act,
                    'v_cmd': info.get('v_cmd', 0.0),
                    'w_cmd': info.get('w_cmd', 0.0)
                })
                
                step_count += 1
                global_step += 1
                
                # Mostrar progreso
                if step_count % 30 == 0:
                    distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
                    print(f"  Paso {step_count}: Dist={distance:.2f}m")
            
            # Detener robot entre waypoints
            self.env._publish_cmd(0.0, 0.0)
            time.sleep(1.0)
            
            # Verificar si alcanzó el waypoint
            final_distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            waypoint_success = final_distance < self.goal_tolerance
            
            if waypoint_success:
                route_stats['waypoints_reached'] += 1
                print(f"  ✅ Waypoint alcanzado en {step_count} pasos (dist: {final_distance:.2f}m)")
            else:
                print(f"  ❌ Waypoint no alcanzado (dist: {final_distance:.2f}m)")
                
                # Preguntar si continuar
                if self.manual_goals:
                    continue_route = input("  ⚠️ Waypoint no alcanzado. ¿Continuar con el siguiente? (y/n): ")
                    if continue_route.lower() != 'y':
                        break
        
        # Detener robot al final
        self.env._publish_cmd(0.0, 0.0)
        
        # Guardar trayectoria completa
        self._save_trajectory_csv(full_trajectory, csv_filename)
        
        # 🆕 NUEVO: Guardar velocidades
        self._save_velocity_csv(velocity_data, velocity_csv_filename)
        
        # Calcular estadísticas finales
        route_stats['total_steps'] = global_step
        route_stats['total_time'] = time.time() - start_time
        route_stats['success_rate'] = route_stats['waypoints_reached'] / len(waypoints) * 100
        
        print(f"\n📊 Estadísticas de la ruta:")
        print(f"   Waypoints alcanzados: {route_stats['waypoints_reached']}/{len(waypoints)}")
        print(f"   Tasa de éxito: {route_stats['success_rate']:.1f}%")
        print(f"   Pasos totales: {route_stats['total_steps']}")
        print(f"   Tiempo total: {route_stats['total_time']:.1f}s")
        print(f"   CSV trayectoria: {csv_filename}")
        print(f"   CSV velocidades: {velocity_csv_filename}")  # 🆕 NUEVO
        
        return route_stats
    
    def test_interactive_navigation(self):
        """
        Modo interactivo: el usuario da objetivos en tiempo real
        """
        print("🎮 Modo navegación interactiva iniciado")
        print("   Comandos: 'g x y' para ir a (x,y), 'stop' para parar, 'quit' para salir")
        
        obs, _ = self.env_wrapper.reset()
        print(f"📍 Robot en posición: ({self.env.pos_x:.2f}, {self.env.pos_y:.2f})")
        
        while True:
            try:
                cmd = input("\n🎯 Comando: ").strip().split()
                
                if not cmd:
                    continue
                elif cmd[0] == 'quit':
                    break
                elif cmd[0] == 'stop':
                    self.env._publish_cmd(0.0, 0.0)
                    print("🛑 Robot detenido")
                elif cmd[0] == 'g' and len(cmd) == 3:
                    goal_x, goal_y = float(cmd[1]), float(cmd[2])
                    print(f"🚀 Navegando a ({goal_x:.2f}, {goal_y:.2f})...")
                    
                    # Configurar objetivo
                    self.env.target_x = goal_x
                    self.env.target_y = goal_y
                    self.env._publish_goal_pose()
                    self.env._publish_goal_marker()
                    
                    # Navegar con timeout
                    max_steps = 200
                    step_count = 0
                    done = False
                    
                    while not done and step_count < max_steps:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = self.env_wrapper.step(action)
                        done = terminated or truncated
                        
                        step_count += 1
                        
                        if step_count % 10 == 0:
                            dist = info.get('distance', 0.0)
                            print(f"  Paso {step_count}: dist={dist:.2f}m")
                        
                        # Permitir interrumpir
                        if step_count % 5 == 0:
                            # Check for user input (non-blocking)
                            pass
                    
                    # Resultado
                    final_dist = info.get('distance', 0.0)
                    if final_dist < self.goal_tolerance:
                        print(f"✅ Objetivo alcanzado! (dist: {final_dist:.2f}m)")
                    else:
                        print(f"⏱️ Timeout (dist: {final_dist:.2f}m)")
                    
                    self.env._publish_cmd(0.0, 0.0)
                    
                else:
                    print("❌ Comando inválido. Usa: 'g x y', 'stop', o 'quit'")
                    
            except ValueError:
                print("❌ Coordenadas inválidas")
            except KeyboardInterrupt:
                print("\n⏸️ Interrumpido")
                break
        
        self.env._publish_cmd(0.0, 0.0)
        print("🏁 Navegación interactiva terminada")
    
    def _get_waypoints_input(self):
        """Obtiene waypoints del usuario."""
        waypoints = []
        print("📍 Ingresa waypoints para la ruta:")
        print("   Formato: x,y (ejemplo: 2.0,1.5)")
        print("   Presiona Enter vacío para terminar")
        
        i = 1
        while True:
            try:
                coord_input = input(f"Waypoint {i}: ").strip()
                if not coord_input:
                    break
                
                x, y = map(float, coord_input.split(','))
                waypoints.append((x, y))
                print(f"  ✅ Waypoint {i}: ({x:.2f}, {y:.2f})")
                i += 1
                
            except ValueError:
                print("❌ Formato inválido. Usa: x,y")
        
        return waypoints
    
    def _save_trajectory_csv(self, trajectory, filename):
        """Guarda la trayectoria en archivo CSV"""
        if not trajectory:
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = trajectory[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory)
    
    # 🆕 NUEVO: Función para guardar velocidades en CSV
    def _save_velocity_csv(self, velocity_data, filename):
        """Guarda los datos de velocidad en archivo CSV separado"""
        if not velocity_data:
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = velocity_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(velocity_data)
    
    def emergency_stop(self):
        """Parada de emergencia del robot."""
        print("🚨 PARADA DE EMERGENCIA")
        self.env._publish_cmd(0.0, 0.0)
        time.sleep(1.0)
    
    def close(self):
        """Cierra el ambiente y ROS2"""
        print("🛑 Deteniendo robot...")
        self.env._publish_cmd(0.0, 0.0)
        time.sleep(1.0)
        self.env_wrapper.close()
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass


def main():
    print("🤖 ROBOT REAL RL TESTER")
    print("========================")
    
    # Solicitar ruta del modelo
    print("\n1. Especifica la ruta del modelo entrenado:")
    model_path = input("Ruta del modelo: ").strip()
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el archivo {model_path}")
        return
    
    # Preguntar modo de operación
    print("\n2. Modo de operación:")
    print("   1. Automático (objetivos aleatorios)")
    print("   2. Manual (tú especificas objetivos)")
    
    mode_choice = input("Selecciona (1-2): ").strip()
    manual_goals = mode_choice == "2"
    
    # Inicializar tester
    try:
        tester = RobotRealTester(model_path, manual_goals=manual_goals)
        print("✅ Tester inicializado correctamente")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        return
    
    try:
        # Menú principal
        while True:
            print("\n" + "="*60)
            print("OPCIONES DE TESTING PARA ROBOT REAL:")
            print("1. Navegar a objetivo único")
            print("2. Navegar ruta de waypoints")
            print("3. Navegación interactiva")
            print("4. Parada de emergencia")
            print("5. Salir")
            
            choice = input("\nSelecciona opción (1-5): ").strip()
            
            if choice == "1":
                print("\n--- OBJETIVO ÚNICO ---")
                if manual_goals:
                    try:
                        goal_x = float(input("Coordenada X del objetivo: "))
                        goal_y = float(input("Coordenada Y del objetivo: "))
                        tester.test_single_goal(goal_x, goal_y)
                    except ValueError:
                        print("❌ Error: Coordenadas inválidas")
                else:
                    tester.test_single_goal()
            
            elif choice == "2":
                print("\n--- RUTA DE WAYPOINTS ---")
                tester.test_waypoint_route()
            
            elif choice == "3":
                print("\n--- NAVEGACIÓN INTERACTIVA ---")
                tester.test_interactive_navigation()
            
            elif choice == "4":
                tester.emergency_stop()
            
            elif choice == "5":
                break
            
            else:
                print("❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n⏸️ Interrumpido por usuario")
        tester.emergency_stop()
    
    finally:
        print("\n🔄 Cerrando...")
        tester.close()
        print("✅ Finalizado")


if __name__ == "__main__":
    main()
