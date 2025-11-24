import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# ===============================
# 1. 系统参数（房间 + 空调的简单模型）
# ===============================
a = 0.05          # 与环境的“散热”系数（越大，越容易被外界拉回去）
b = 0.1           # 空调对温度的作用强度
T_env = 30.0      # 外界温度（比如夏天 30℃）

# ===============================
# 2. MPC 参数
# ===============================
dt = 1.0          # 采样时间（1 个时间步）
N = 20            # 预测步数（MPC 每次向前看 20 步）

# 控制量约束（空调功率）
u_min = -2.0      # 最强制冷
u_max =  2.0      # 最强制热

# 温度安全范围约束
T_min = 18.0      # 最低不能低于 18℃
T_max = 28.0      # 最高不能高于 28℃

# 成本函数权重：温度误差 + 控制量
lambda_u = 0.01   # 控制量惩罚系数（越大，空调越“温柔”）

# ===============================
# 3. 仿真设置
# ===============================
K = 60            # 总共仿真步数（相当于 60 秒 / 60 个时间单位）
T0 = 26.0         # 初始室温

# 构造一个简单的目标温度轨迹（可以随便改）
T_ref_traj = np.zeros(K)
T_ref_traj[:20] = 24.0   # 前 20 步希望 24℃
T_ref_traj[20:40] = 22.0 # 中间 20 步希望 22℃
T_ref_traj[40:] = 25.0   # 最后 20 步希望 25℃

# 用来记录仿真结果
T_hist = np.zeros(K+1)
u_hist = np.zeros(K)

T_hist[0] = T0

# ===============================
# 4. MPC 主循环（核心）
# ===============================
for k in range(K):
    # 当前状态（当前温度）
    T_now = T_hist[k]
    # 当前参考温度（简单处理：未来 N 步都用当前这个目标温度）
    T_ref = T_ref_traj[k]

    # ----- 定义优化变量 -----
    # 未来 N 步的控制量 u[0..N-1]
    u = cp.Variable(N)
    # 未来 N+1 个温度 T[0..N]
    T = cp.Variable(N + 1)

    # ----- 约束集合 -----
    constraints = []

    # 初始条件：T[0] = 当前温度
    constraints += [T[0] == T_now]

    # 系统动态约束：T[i+1] = T[i] + a*(T_env - T[i]) + b*u[i]
    for i in range(N):
        constraints += [
            T[i+1] == T[i] + a * (T_env - T[i]) + b * u[i]
        ]
        # 控制量约束
        constraints += [
            u[i] >= u_min,
            u[i] <= u_max
        ]
        # 温度约束
        constraints += [
            T[i+1] >= T_min,
            T[i+1] <= T_max
        ]

    # ----- 成本函数（目标） -----
    cost = 0
    for i in range(1, N+1):
        # 温度误差平方
        cost += cp.square(T[i] - T_ref)
    # 控制量平方（不希望空调忽大忽小）
    for i in range(N):
        cost += lambda_u * cp.square(u[i])

    # 构建优化问题：min cost s.t. constraints
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # 求解（cvxpy 会调用合适的 QP 求解器）
    problem.solve(solver=cp.OSQP, warm_start=True)

    if u.value is None:
        # 如果优化失败（不太可能），就用 0 控制
        u_opt = 0.0
        print(f"Step {k}: MPC 求解失败，使用 u=0 作为控制。")
    else:
        # MPC 只执行未来 N 步中的第 1 步
        u_opt = u.value[0]

    # 用真实模型推进一步（这里直接用同一个模型）
    T_next = T_now + a * (T_env - T_now) + b * u_opt

    # 记录
    u_hist[k] = u_opt
    T_hist[k+1] = T_next

# ===============================
# 5. 画图看结果
# ===============================
time = np.arange(K+1)

plt.figure(figsize=(10, 6))

# 温度曲线
plt.subplot(2, 1, 1)
plt.plot(time, T_hist, label="室温 T")
plt.plot(time[:-1], T_ref_traj, "--", label="目标温度 T_ref")
plt.axhline(T_min, linestyle=":", label="T_min")
plt.axhline(T_max, linestyle=":", label="T_max")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)

# 控制量曲线
plt.subplot(2, 1, 2)
plt.step(time[:-1], u_hist, where="post", label="空调功率 u")
plt.axhline(u_min, linestyle=":", label="u_min")
plt.axhline(u_max, linestyle=":", label="u_max")
plt.xlabel("Time step")
plt.ylabel("u")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
