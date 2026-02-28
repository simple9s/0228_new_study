import psutil
import os
import time
import signal
import sys

filename = ['main.py','train.py','train_hybrid.py','inference.py']

class PythonProcessMonitor:
    def __init__(self, check_interval=30, shutdown_delay=60):
        self.check_interval = check_interval
        self.shutdown_delay = shutdown_delay
        self.current_pid = os.getpid()
        self.running = True
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print(f"\n接收到信号 {signum}，正在停止监控...")
        self.running = False
    
    def check_specific_python_processes(self):
        """
        检查是否有特定的Python进程在运行
        返回：True-有特定进程运行，False-没有特定进程
        """
        current_pid = os.getpid()
        found_target_process = False
        
        print("🔍 正在检查进程...")
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # 排除当前进程
                if proc.info['pid'] == current_pid:
                    continue
                
                # 检查是否是Python进程
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    
                    # 调试信息：显示所有Python进程的命令行
                    # if 'python' in cmdline:
                    #     print(f"  📝 发现Python进程 PID {proc.info['pid']}: {proc.info['cmdline']}")
                    
                    # 检查是否包含 python main.py
                    for i in filename:
                        if ('python' in cmdline and i in cmdline) or \
                        ('python' in cmdline and any(i in arg for arg in proc.info['cmdline'])):
                        
                            # print(f"  ✅ 找到目标进程: PID {proc.info['pid']}")
                            # print(f"     命令行: {' '.join(proc.info['cmdline'])}")
                            found_target_process = True
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        return found_target_process
    
    def execute_shutdown(self):
        """执行关机"""
        print(f"⚠️ 没有检测到 'python main.py' 进程，系统将在 {self.shutdown_delay} 秒后关机！")
        print("按 Ctrl+C 取消关机")
        
        try:
            for i in range(self.shutdown_delay, 0, -1):
                if not self.running:
                    print("\n取消关机...")
                    return
                print(f"\r关机倒计时: {i}秒 (Ctrl+C取消)", end="", flush=True)
                time.sleep(1)
            print("\n再次检测 避免误关机...")
            if self.check_specific_python_processes():
                print("✅ 检测到 'python main.py' 进程，继续运行")
                return
            print("\n正在关机...")
            # 这里可以添加实际的关机命令
            if os.name == 'nt':
                os.system("shutdown /s /t 0")
            else:
                os.system("shutdown -h now")
            
        except KeyboardInterrupt:
            print("\n\n关机已取消")
            self.running = False
    
    def start_monitoring(self):
        """开始监控"""
        print(f"🐍 Python进程监控器已启动 (PID: {self.current_pid})")
        print("🎯 监控目标: 'python main.py' 进程")
        print(f"📊 检查间隔: {self.check_interval}秒")
        print("按 Ctrl+C 停止监控\n")
        
        check_count = 0
        
        while self.running:
            check_count += 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] 第 {check_count} 次检查...")
            
            if self.check_specific_python_processes():
                print("✅ 检测到 'python main.py' 进程，继续运行")
            else:
                print("❌ 未检测到 'python main.py' 进程")
                self.execute_shutdown()
            
            print("-" * 50)
            
            # 等待下一次检查
            if self.running:
                for i in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
        
        print("监控器已停止")

def main():
    CHECK_INTERVAL = 30
    SHUTDOWN_DELAY = 60
    
    monitor = PythonProcessMonitor(
        check_interval=CHECK_INTERVAL,
        shutdown_delay=SHUTDOWN_DELAY
    )
    
    try:
        monitor.start_monitoring()
    except Exception as e:
        print(f"监控器出错: {e}")

if __name__ == "__main__":
    main()