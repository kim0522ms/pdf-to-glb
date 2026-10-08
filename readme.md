# PDF to GLB

U3D가 포함된 PDF를 GLB로 변환합니다. 단색 CLOD 메시를 지원합니다.

## 설치

Python 3.11 이상이 필요합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 실행

```sh
python pdf_to_glb.py input.pdf output.glb
```

`.u3d` 파일도 입력할 수 있습니다. 형상, 색상, 부품 배치와 원본 좌표를 유지합니다.
색상은 sRGB로 가정하며, 광택과 반사는 PBR 재질로 근사합니다.

## 테스트

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q --cov=pdf_to_glb --cov-branch
```

U3D 디코더의 출처와 라이선스는 `THIRD_PARTY_NOTICE.txt`를 참고하십시오.
