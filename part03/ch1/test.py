import requests
import sys

print(f"▶ 현재 구동 중인 파이썬 버전: {sys.version.split()[0]}")
print(f"▶ 외부 패키지 인식 성공 (requests 버전): {requests.__version__}")