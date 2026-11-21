"""V1物理环境最小运行示例。请在项目根目录执行：python examples/run_demo.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Env import Env, ActionPlan


def main():
    env = Env()
    snapshot = env.reset()
    print(f"初始化完成：t={snapshot.time}s，阶段={snapshot.phase}，列车数={len(snapshot.trains)}")
    print("观测中断：", snapshot.disruption)
    for i in range(3):
        result = env.step(ActionPlan())
        print(f"第{i + 1}轮：t={result.snapshot.time}s，等待人秒={result.metrics['window_wait_pax_sec']:.0f}，"
              f"站台人数={int(result.snapshot.platform.sum())}，车内人数={int(result.snapshot.onboard.sum())}，"
              f"守恒差={result.metrics['conservation_gap']:.0f}")


if __name__ == "__main__":
    main()
