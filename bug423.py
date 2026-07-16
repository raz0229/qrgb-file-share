import qrcode
from qrcode.util import QRData, MODE_8BIT_BYTE
from qrcode.constants import ERROR_CORRECT_Q

# Version 5 + Q = 60 bytes capacity
data = b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x16Hello from kakaworld!!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
assert len(data) == 60  # exactly at capacity

qr = qrcode.QRCode(version=5, error_correction=ERROR_CORRECT_Q)
qr.add_data(QRData(data, mode=MODE_8BIT_BYTE))
qr.make(fit=False)  # ValueError: glog(0)