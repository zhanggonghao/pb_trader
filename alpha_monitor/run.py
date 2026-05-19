"""
启动脚本 - 智能定时启停

功能：
  1. 判断当日是否为交易日，非交易日不启动数据刷新
  2. 交易日 9:30 启动数据刷新，15:00 停止数据刷新
  3. 启动前自动杀掉上次的进程
  4. 关闭数据刷新后，网页和API依然可访问（返回最后缓存的数据）
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_monitor.config import CONFIG
from alpha_monitor import alpha_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("alpha_monitor")

STATIC_DIR = Path(__file__).parent / "static"
latest_data = {"timestamp": "", "trading_time": False, "market_closed": True}
data_lock = asyncio.Lock()
refresh_enabled = False  # 控制数据刷新是否开启

MIME_MAP = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
}


# ==================== 交易日/时间判断 ====================

def is_trading_day() -> bool:
    """判断今天是不是交易日（用米筐）"""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        import rqdatac
        return rqdatac.is_trading_date(today)
    except Exception as e:
        logger.warning(f"交易日判断失败: {e}，默认否")
        return False


def current_time_in_range(start_hour: int, start_min: int, end_hour: int, end_min: int) -> bool:
    """判断当前时间是否在指定范围内"""
    now = datetime.now().time()
    start = dtime(start_hour, start_min, 0)
    end = dtime(end_hour, end_min, 0)
    return start <= now <= end


def should_refresh() -> bool:
    """判断当前是否应刷新数据（交易日 9:30 ~ 15:00）"""
    if not is_trading_day():
        return False
    return current_time_in_range(9, 30, 15, 0)


def kill_existing_process():
    """杀掉上次遗留的 run.py 进程"""
    this_pid = os.getpid()
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if proc.info['pid'] == this_pid:
                    continue
                cmdline = proc.info.get('cmdline')
                if cmdline and 'run.py' in ' '.join(cmdline).lower():
                    logger.warning(f"杀掉旧进程 PID={proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # 没有 psutil，用 wmic 方式
        try:
            result = subprocess.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'processid,commandline', '/format:csv'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n')[1:]:
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    cmd = parts[-2] if len(parts) >= 3 else ''
                    pid = parts[-1].strip()
                    if pid and pid.isdigit() and int(pid) != this_pid and 'run.py' in cmd.lower():
                        logger.warning(f"杀掉旧进程 PID={pid}")
                        os.kill(int(pid), signal.SIGTERM)
        except Exception as e:
            logger.warning(f"杀掉旧进程失败: {e}")


# ==================== HTTP 服务 ====================

async def send_response(writer, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
    status_map = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}
    header = (
        f"HTTP/1.1 {status} {status_map.get(status, 'Unknown')}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        f"Connection: keep-alive\r\n"
        f"\r\n"
    ).encode('utf-8')
    writer.write(header + body)
    await writer.drain()


async def send_json(writer, data):
    body = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
    await send_response(writer, 200, body, "application/json; charset=utf-8")


async def handle_request(reader, writer):
    try:
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = await reader.read(4096)
            if not chunk:
                break
            raw += chunk
            if len(raw) > 65536:
                break
        if not raw:
            writer.close()
            return
        text = raw.decode('utf-8', errors='replace')
        lines = text.split('\r\n')
        if not lines:
            writer.close()
            return
        parts = lines[0].split(' ')
        if len(parts) < 2:
            writer.close()
            return
        _, path = parts[0], parts[1]
        parsed = urlparse(path)
        clean = parsed.path

        if clean == '/api/data':
            async with data_lock:
                await send_json(writer, latest_data)
            return
        if clean == '/api/health':
            await send_json(writer, {
                "status": "ok",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trading": refresh_enabled,
                "trading_day": is_trading_day()
            })
            return

        if clean == '/':
            clean = '/index.html'
        fp = STATIC_DIR / clean.lstrip('/')
        if fp.exists() and fp.is_file():
            content = fp.read_bytes()
            ct = MIME_MAP.get(fp.suffix.lower(), 'application/octet-stream')
            await send_response(writer, 200, content, ct)
        else:
            await send_response(writer, 404, b"Not Found")
    except Exception as e:
        logger.error(f"请求处理错误: {e}")
        try:
            await send_response(writer, 500, b"Internal Server Error")
        except Exception:
            pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


# ==================== 数据刷新循环 ====================

async def data_refresh_loop():
    """后台数据刷新循环（仅在 refresh_enabled 时执行）"""
    global latest_data
    while True:
        if refresh_enabled:
            try:
                data = alpha_engine.collect_all_data()
                # 标记数据为交易时段
                data["market_closed"] = False
                async with data_lock:
                    latest_data = data
            except Exception as e:
                logger.error(f"数据刷新失败: {e}")
        else:
            # 非交易时段——保持 latest_data 不变，只更新时间戳和收盘标记
            async with data_lock:
                latest_data["market_closed"] = True
                latest_data["trading_time"] = False
        await asyncio.sleep(CONFIG.server.refresh_interval)


# ==================== 定时启停调度 ====================

async def scheduler_loop():
    """每30秒检查一次是否需要启停数据刷新"""
    global refresh_enabled
    prev_state = None
    while True:
        new_state = should_refresh()

        if new_state != prev_state:
            if new_state:
                logger.info("========== 交易时段开始，启动数据刷新 ==========")
                async with data_lock:
                    # 启动时立即刷新一次数据
                    try:
                        data = alpha_engine.collect_all_data()
                        data["market_closed"] = False
                        latest_data = data
                        logger.info("首次数据刷新完成")
                    except Exception as e:
                        logger.error(f"首次数据刷新失败: {e}")
            else:
                if prev_state is True:
                    logger.info("========== 交易时段结束，停止数据刷新 ==========")
                    async with data_lock:
                        latest_data["market_closed"] = True
                        latest_data["trading_time"] = False
                else:
                    logger.debug("非交易时段，等待...")

            refresh_enabled = new_state
            prev_state = new_state

        await asyncio.sleep(30)


# ==================== 主入口 ====================

async def main():
    """启动HTTP服务和调度器，数据刷新由调度器控制"""
    global refresh_enabled

    logger.info("=" * 50)
    logger.info("[Alpha] 实时Alpha超额监控系统 v2.0 (智能定时版)")
    logger.info(f"[Server] 监控地址: http://{CONFIG.server.host}:{CONFIG.server.port}")
    logger.info(f"[Config] 刷新间隔: {CONFIG.server.refresh_interval}秒")
    logger.info(f"[Config] 基准指数: {CONFIG.benchmark.name}")
    logger.info(f"[Config] 监控账户: {[a.name for a in CONFIG.accounts]}")
    logger.info(f"[Config] 期货品种: {[f.name for f in CONFIG.futures]}")

    # 判断当前状态
    today_is_trading = is_trading_day()
    now = datetime.now()
    if today_is_trading:
        if current_time_in_range(9, 30, 15, 0):
            logger.info("[Status] 今日交易日 · 交易时段内，立即启动数据刷新")
        elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
            logger.info("[Status] 今日交易日 · 等待 9:30 开市")
        else:
            logger.info("[Status] 今日交易日 · 已收盘，仅提供静态页面")
    else:
        logger.info("[Status] 非交易日，仅提供静态页面")

    logger.info("[Note] 网页始终可访问，数据仅在交易时段更新")
    logger.info(f" http://{CONFIG.server.host}:{CONFIG.server.port}")
    logger.info("=" * 50)

    # 初始化数据源
    from alpha_monitor import market_data as mdata
    mdata.init()

    # 首次数据尝试
    global latest_data
    try:
        data = alpha_engine.collect_all_data()
        async with data_lock:
            latest_data = data
            latest_data["market_closed"] = not should_refresh()
        logger.info("首次数据获取完成")
    except Exception as e:
        logger.warning(f"首次数据获取失败（非交易时段可能无数据）: {e}")

    # 如果当前在交易时段内，开启刷新
    if should_refresh():
        refresh_enabled = True
        logger.info("数据刷新已开启")

    # 启动调度器
    asyncio.create_task(scheduler_loop())

    # 启动数据刷新循环
    asyncio.create_task(data_refresh_loop())

    # 启动HTTP服务
    server = await asyncio.start_server(
        handle_request, CONFIG.server.host, CONFIG.server.port
    )
    addr = server.sockets[0].getsockname()
    logger.info(f"[OK] 服务已启动: http://{addr[0]}:{addr[1]}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    # 启动前杀掉旧进程
    kill_existing_process()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务已停止")
