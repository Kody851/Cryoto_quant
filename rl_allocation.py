import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# 设置随机种子，保证结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 定义经验回放缓冲区
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        super(DQN, self).__init__()
        layers = []
        input_size = state_size
        
        # 创建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(input_size, action_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# 定义股票交易环境
class StockTradingEnv:
    def __init__(self, stock_data: pd.DataFrame, window_size: int = 10, initial_capital: float = 10000.0):
        """
        初始化股票交易环境
        :param stock_data: 包含多只股票价格的DataFrame，索引为日期，列包含每只股票的收盘价
        :param window_size: 用于观察的历史窗口大小
        :param initial_capital: 初始资金
        """
        self.stock_data = stock_data
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.n_stocks = stock_data.shape[1]  # 股票数量
        
        # 计算每日收益率
        self.returns = stock_data.pct_change().fillna(0)
        
        # 重置环境
        self.reset()
        
    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.shares_held = np.zeros(self.n_stocks)  # 持有的每只股票数量
        self.total_assets = self.initial_capital
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 最近window_size天的收益率
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        returns_window = self.returns.iloc[start_idx:end_idx].values.flatten()
        
        # 当前持有的股票比例
        holdings_ratio = self.shares_held * self.stock_data.iloc[self.current_step].values / self.total_assets
        holdings_ratio = np.nan_to_num(holdings_ratio)  # 处理可能的NaN
        
        # 合并状态
        state = np.concatenate([returns_window, holdings_ratio])
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        执行动作并返回新状态、奖励和是否结束
        :param action: 资产配置比例，长度为n_stocks，总和为1
        :return: 新状态、奖励、是否结束
        """
        # 确保动作是有效的资产配置比例
        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)  # 归一化
        else:
            action = np.zeros_like(action)
        
        # 记录当前总资产作为基准
        prev_assets = self.total_assets
        
        # 获取当前价格
        current_prices = self.stock_data.iloc[self.current_step].values
        
        # 根据目标配置比例调整持仓
        target_values = action * self.total_assets
        target_shares = np.round(target_values / current_prices).astype(int)
        target_shares = np.maximum(target_shares, 0)  # 确保不持有负股票
        
        # 计算需要交易的股票数量
        shares_to_trade = target_shares - self.shares_held
        
        # 执行交易
        for i in range(self.n_stocks):
            if shares_to_trade[i] > 0:  # 买入
                cost = shares_to_trade[i] * current_prices[i]
                if cost <= self.capital:
                    self.capital -= cost
                    self.shares_held[i] = target_shares[i]
            elif shares_to_trade[i] < 0:  # 卖出
                revenue = -shares_to_trade[i] * current_prices[i]
                self.capital += revenue
                self.shares_held[i] = target_shares[i]
        
        # 移动到下一步
        self.current_step += 1
        
        # 计算新的总资产
        portfolio_value = np.sum(self.shares_held * self.stock_data.iloc[self.current_step].values)
        self.total_assets = self.capital + portfolio_value
        
        # 计算奖励（资产变化率）
        reward = (self.total_assets - prev_assets) / prev_assets
        
        # 检查是否结束
        done = self.current_step >= len(self.stock_data) - 1
        
        # 获取新状态
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done
    
    def get_total_assets(self) -> float:
        """返回当前总资产"""
        return self.total_assets

# 定义智能体
class Agent:
    def __init__(self, state_size: int, action_size: int, 
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 lr: float = 0.001, batch_size: int = 64, buffer_capacity: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # 创建策略网络和目标网络
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_capacity)
        
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        根据当前状态选择动作
        :param state: 当前状态
        :param deterministic: 是否确定性选择（测试时用）
        :return: 动作（资产配置比例）
        """
        if not deterministic and random.random() < self.epsilon:
            # 随机探索：生成随机的资产配置比例
            action = np.random.rand(self.action_size)
            action = action / np.sum(action)  # 归一化
            return action
        else:
            # 贪婪选择：使用策略网络
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                # 将Q值转换为配置比例（确保总和为1）
                action = torch.softmax(q_values, dim=1).squeeze().numpy()
                return action
    
    def replay(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样经验
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 转换为张量
        states = torch.FloatTensor(batch.state)
        actions = torch.FloatTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.policy_net(states)
        # 这里我们需要处理连续动作空间的Q值估计
        # 简化处理：将动作与Q值做点积，得到该动作的Q值估计
        current_q_values = torch.sum(current_q * actions, dim=1, keepdim=True)
        
        # 计算目标Q值
        next_q = self.target_net(next_states)
        max_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 计算损失并优化
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 生成模拟股票数据
def generate_stock_data(n_stocks: int = 5, n_days: int = 1000, start_date: str = '2020-01-01') -> pd.DataFrame:
    """生成模拟的股票价格数据"""
    dates = pd.date_range(start_date, periods=n_days)
    data = {}
    
    for i in range(n_stocks):
        # 生成带有趋势和噪声的价格序列
        trend = np.linspace(100, 100 + random.uniform(-20, 80), n_days)
        noise = np.random.normal(0, 2, n_days)
        # 添加一些周期性波动
        seasonality = 5 * np.sin(np.linspace(0, 10 * np.pi, n_days))
        price = trend + noise + seasonality
        price = np.maximum(price, 1)  # 确保价格为正
        data[f'Stock_{i+1}'] = price
    
    return pd.DataFrame(data, index=dates)

# 训练函数
def train(agent: Agent, env: StockTradingEnv, episodes: int = 50, 
          update_target_every: int = 10, max_steps: int = 1000) -> List[float]:
    """训练智能体"""
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)
            
            # 学习
            loss = agent.replay()
            
            # 更新状态和奖励
            state = next_state
            total_reward += reward
            steps += 1
            
            # 打印进度
            if steps % 100 == 0:
                print(f"Episode {episode+1}/{episodes}, Step {steps}, Total Assets: {env.get_total_assets():.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # 定期更新目标网络
        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()
            print(f"更新目标网络 (Episode {episode+1})")
        
        # 记录总奖励
        total_rewards.append(total_reward)
        final_assets = env.get_total_assets()
        print(f"Episode {episode+1}/{episodes} 完成. 总奖励: {total_reward:.4f}, 最终资产: {final_assets:.2f}, 收益率: {(final_assets / env.initial_capital - 1) * 100:.2f}%")
    
    return total_rewards

# 测试函数
def test(agent: Agent, env: StockTradingEnv, deterministic: bool = True) -> Tuple[float, List[float]]:
    """测试训练好的智能体"""
    state = env.reset()
    total_reward = 0.0
    done = False
    assets_history = [env.get_total_assets()]
    actions_history = []
    
    while not done:
        # 选择动作（确定性）
        action = agent.act(state, deterministic=deterministic)
        actions_history.append(action)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        
        # 更新状态和奖励
        state = next_state
        total_reward += reward
        
        # 记录资产变化
        assets_history.append(env.get_total_assets())
    
    final_assets = env.get_total_assets()
    print(f"测试完成. 最终资产: {final_assets:.2f}, 收益率: {(final_assets / env.initial_capital - 1) * 100:.2f}%")
    
    return total_reward, assets_history, actions_history

# 主函数
def main():
    # 生成模拟股票数据（5只股票，1000天）
    n_stocks = 5
    n_days = 1000
    stock_data = generate_stock_data(n_stocks, n_days)
    
    # 划分训练集和测试集（8:2）
    split_idx = int(0.8 * n_days)
    train_data = stock_data.iloc[:split_idx]
    test_data = stock_data.iloc[split_idx:]
    
    # 窗口大小
    window_size = 10
    
    # 创建环境
    train_env = StockTradingEnv(train_data, window_size=window_size)
    test_env = StockTradingEnv(test_data, window_size=window_size)
    
    # 状态大小 = 窗口大小*股票数量 + 股票数量（持有比例）
    state_size = window_size * n_stocks + n_stocks
    action_size = n_stocks
    
    # 创建智能体
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        lr=0.0001,
        batch_size=64,
        buffer_capacity=50000
    )
    
    # 训练智能体
    print("开始训练...")
    training_rewards = train(agent, train_env, episodes=30, update_target_every=5)
    
    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(training_rewards)
    plt.title('训练过程中的总奖励')
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.grid(True)
    plt.show()
    
    # 测试智能体
    print("\n开始测试...")
    test_reward, assets_history, actions_history = test(agent, test_env)
    
    # 绘制资产变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(assets_history)
    plt.title('测试期间的资产变化')
    plt.xlabel('天数')
    plt.ylabel('总资产')
    plt.grid(True)
    plt.show()
    
    # 绘制最后10天的资产配置比例
    if len(actions_history) >= 10:
        last_10_actions = np.array(actions_history[-10:])
        days = range(len(actions_history)-9, len(actions_history)+1)
        
        plt.figure(figsize=(12, 6))
        for i in range(n_stocks):
            plt.plot(days, last_10_actions[:, i], label=f'Stock {i+1}')
        
        plt.title('最后10天的资产配置比例')
        plt.xlabel('天数')
        plt.ylabel('配置比例')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
    
