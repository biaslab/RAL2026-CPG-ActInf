import asyncio
from bleak import BleakClient

RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

async def main(addr):
    async with BleakClient(addr) as c:
        await c.start_notify(TX, lambda _, d: print(d.decode(errors="replace"), end=""))
        await c.write_gatt_char(RX, b"khi\n")
        await asyncio.sleep(2)

asyncio.run(main("XX:XX:XX:XX:XX:XX"))