from pydantic import BaseModel


class SupportedDevice(BaseModel):
    name: str
    mem: int
    bandwidth: str


all_supported_device = [
    SupportedDevice(name="NV-A100", mem=40, bandwidth="1555GB/s"),
    SupportedDevice(name="NV-4090", mem=24, bandwidth="1008GB/s"),
    SupportedDevice(name="NV-A30", mem=24, bandwidth="933GB/s"),
    SupportedDevice(name="NV-3090", mem=24, bandwidth="936GB/s"),
    SupportedDevice(name="EF-S60", mem=48, bandwidth="650GB/s"),
    SupportedDevice(name="NV-4060", mem=8, bandwidth="288GB/s"),
    SupportedDevice(name="NV-P40", mem=24, bandwidth="346GB/s"),
    SupportedDevice(name="NV-3060", mem=12, bandwidth="360GB/s"),
    SupportedDevice(name="CPU", mem=99999, bandwidth="0GB"),
]
