import websockets
import asyncio
import json
import sys

async def get_btc_price():
    # Binance的WebSocket端点，用于BTC/USDT的实时价格
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
    
    async with websockets.connect(url) as websocket:
        print("已连接到Binance WebSocket，正在获取BTC价格...")
        print("按Ctrl+C停止程序")
        
        try:
            while True:
                # 接收WebSocket消息
                data = await websocket.recv()
                # 解析JSON数据
                ticker = json.loads(data)
                
                # 提取所需的价格信息
                symbol = ticker['s']
                last_price = ticker['c']
                high_price = ticker['h']
                low_price = ticker['l']
                volume = ticker['v']
                
                # 打印价格信息
                output = f"\r{symbol} 最新价格: {last_price} USDT | 24h最高: {high_price} | 24h最低: {low_price} | 24h成交量: {volume}"
                print(output, end='')
                sys.stdout.flush()
                
        except KeyboardInterrupt:
            print("\n程序已停止")
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    # 检查是否已安装websockets库，如果没有则提示安装
    try:
        import websockets
    except ImportError:
        print("请先安装websockets库：pip install websockets")
        return
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("警告：推荐使用Python 3.7或更高版本以获得最佳体验")
    
    # 处理事件循环问题
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # 如果已有运行的事件循环，创建任务并运行
            task = loop.create_task(get_btc_price())
            loop.run_until_complete(task)
        else:
            # 如果事件循环存在但未运行，使用它
            loop.run_until_complete(get_btc_price())
    except RuntimeError:
        # 如果没有运行的事件循环，使用asyncio.run()
        asyncio.run(get_btc_price())

if __name__ == "__main__":
    main()
