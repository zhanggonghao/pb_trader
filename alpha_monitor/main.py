"""
实时Alpha超额监控系统 - 主服务
纯 Python asyncio 实现，HTTP轮询模式
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import CONFIG
from . import market_data as mdata
from . import alpha_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("alpha_monitor")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

latest_data = {"timestamp": "", "trading_time": False}
data_lock = asyncio.Lock()

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
                "trading": mdata.is_trading_time()
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


async def data_refresh_loop():
    """后台数据刷新循环"""
    global latest_data
    while True:
        try:
            data = alpha_engine.collect_all_data()
            async with data_lock:
                latest_data = data
        except Exception as e:
            logger.error(f"数据刷新失败: {e}")
        await asyncio.sleep(CONFIG.server.refresh_interval)


async def main():
    logger.info("=" * 50)
    logger.info("[Data] 实时Alpha超额监控系统 v1.0")
    logger.info(f"[Refresh] 刷新间隔: {CONFIG.server.refresh_interval}秒")
    logger.info(f"[Benchmark] 基准: {CONFIG.benchmark.name} ({CONFIG.benchmark.code})")
    for a in CONFIG.accounts:
        logger.info(f"[Account] {a.name} ({a.short_name})")
    logger.info(f"[Futures] 期货: {[f.name for f in CONFIG.futures]}")
    logger.info(f" http://{CONFIG.server.host}:{CONFIG.server.port}")
    logger.info("=" * 50)

    mdata.init()
    global latest_data
    latest_data = alpha_engine.collect_all_data()

    asyncio.create_task(data_refresh_loop())

    server = await asyncio.start_server(
        handle_request, CONFIG.server.host, CONFIG.server.port
    )
    addr = server.sockets[0].getsockname()
    logger.info(f"[OK] 服务已启动: http://{addr[0]}:{addr[1]}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务已停止")
