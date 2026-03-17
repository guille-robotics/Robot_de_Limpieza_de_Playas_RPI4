# 🤖 Robot Autónomo de Limpieza de Playas (Implementación con RL - RPi4)

<p align="center">
  <img src="assets/robot.gif" width="600"/>
</p>

Este repositorio contiene la implementación real del robot autónomo de limpieza de playas ejecutándose en una Raspberry Pi 4, utilizando modelos entrenados mediante Aprendizaje por Refuerzo (Reinforcement Learning).

Este sistema permite ejecutar directamente en el robot modelos previamente entrenados, integrándose con ROS 2 para la percepción, control y navegación.

---

## 🧠 Descripción

Este proyecto corresponde a la ejecución en hardware real de un robot móvil autónomo que navega utilizando modelos de aprendizaje por refuerzo.

El sistema utiliza:

- 📡 Datos de LiDAR
- 📍 Odometría del robot
- 🧠 Modelos RL entrenados (PPO, DQN, A2C, etc.)
- ⚙️ ROS 2 para comunicación y control

A diferencia de la simulación, este repositorio está enfocado en la **inferencia en tiempo real sobre el robot físico**.

---

## 🔗 Relación con otros repositorios

Este repositorio trabaja en conjunto con:

- Sistema base ROS 2 del robot (sensores, drivers, bringup)
- Repositorio de navegación reactiva (Braitenberg)

📌 Flujo de ejecución típico:

1. Se inicia el sistema base del robot (ROS 2)
2. Luego se ejecuta este repositorio para activar el control mediante RL

---

## 🧩 Estructura del proyecto

Dentro de:

```bash
src/robot_tesis_rl/robot_tesis_rl/
```

Se encuentran los distintos modelos:

```bash
A2C/
DQN/
PPO/
PPOMask/
```

Cada carpeta contiene:

- `ambiente_robot_real.py` → definición del entorno real  
- `robot_env.py` → interfaz del entorno tipo Gym  
- `test_robot_real.py` → ejecución del robot  
- `test_robot_real_con_velocidad.py` → ejecución con registro de velocidades  
- `best_model_*.zip` → modelo entrenado  

---

## 🧠 Modelos de Aprendizaje por Refuerzo

Se incluyen distintas implementaciones:

- PPO  
- DQN  
- A2C  
- PPO con máscara de acciones  

Cada modelo puede ser ejecutado de forma independiente.

---

## ▶️ Ejecución del sistema

### 1. Iniciar sistema base del robot

Primero se debe ejecutar el sistema ROS 2 del robot (desde el otro repositorio):

- LiDAR  
- Odometría  
- Control de motores  

---

### 2. Ejecutar modelo RL

Ejemplo con PPO:

```bash
python3 src/robot_tesis_rl/robot_tesis_rl/PPO/test_robot_real.py
```

---

### 3. Ejecutar con registro de velocidad

```bash
python3 src/robot_tesis_rl/robot_tesis_rl/PPO/test_robot_real_con_velocidad.py
```

Esto permite guardar datos de velocidad para análisis posterior.

---

## ⚙️ Funcionamiento

El sistema realiza:

1. Lectura de sensores (LiDAR + odometría)
2. Construcción del estado del entorno
3. Evaluación del modelo RL
4. Generación de acción (`cmd_vel`)
5. Movimiento del robot

Todo en tiempo real sobre la Raspberry Pi 4.

---

## 📊 Datos recolectados

El sistema permite registrar:

- Velocidad lineal  
- Velocidad angular  
- Comportamiento del robot  

Estos datos pueden utilizarse para análisis y validación.

---

## 📁 Estructura general

```bash
robot_tesis_rl/
├── A2C/
├── DQN/
├── PPO/
├── PPOMask/
├── robot_env.py
├── ambiente_robot_real.py
└── test scripts
```

---

## 🎯 Objetivo

- Ejecutar modelos RL en robot real  
- Validar desempeño fuera de simulación  
- Comparar distintos algoritmos  
- Analizar comportamiento en entornos reales  

---

## 🚧 Estado del proyecto

En desarrollo y validación en robot real.

---

## 👨‍💻 Autor

Guillermo Cid Ampuero  

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia incluida en el repositorio.