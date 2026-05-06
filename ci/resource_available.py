#resource_available
import psutil
import shutil

def gb(x):
    return round(x / (1024**3), 2)

print("\n" + "=" * 50)
print("SYSTEM INFO (BEFORE TESTS)")
print("=" * 50)

cpu = psutil.cpu_count(logical=True)
mem = psutil.virtual_memory()
disk = shutil.disk_usage("/")

print(f"CPU cores        : {cpu}")
print(f"RAM total        : {gb(mem.total)} GB")
print(f"RAM available    : {gb(mem.available)} GB")
print(f"Disk total       : {gb(disk.total)} GB")
print(f"Disk free        : {gb(disk.free)} GB")

print("=" * 50)