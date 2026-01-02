# SK Hynix Y1 Cluster - IRFM Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Proprietary-yellow.svg)](LICENSE)

**Industrial Resources Flow Management System**  
SK Ecoplant 구축 | TJLABS 시스템

실시간 자원 흐름 모니터링 및 예측 대시보드

---

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [배포](#배포)
- [문제 해결](#문제-해결)
- [문서](#문서)

---

## 🎯 개요

DeepCon IRFM Dashboard는 SK Hynix Y1 Cluster의 산업 자원 흐름을 실시간으로 모니터링하고 AI 기반 예측을 제공하는 웹 대시보드입니다.

### 주요 특징

- 📊 **실시간 모니터링**: T-Ward Type31/41, 모바일 디바이스 위치 추적
- 🤖 **AI 예측**: Transformer 기반 리스크 예측
- 📈 **시각화**: Interactive Plotly 차트 및 애니메이션
- 🎮 **시뮬레이션**: 시나리오 기반 리스크 시뮬레이션
- 🚀 **성능 최적화**: 효율적인 캐싱 및 데이터 처리

---

## ✨ 주요 기능

### 1. Overview (개요)
- 전체 시스템 요약 통계
- 실시간 디바이스 상태
- 시간대별 활동 분포

### 2. T-Ward Type31 (장비 분석)
- 장비 가동률 모니터링
- 구역별 분포 분석
- 체류 시간 분석
- 이동 경로 히트맵

### 3. T-Ward Type41 (작업자 분석)
- 작업자 활동 추적
- 활성/비활성 상태 모니터링
- 구역별 인원 현황
- 실시간 위치 애니메이션

### 4. MobilePhone (모바일 분석)
- Android/iPhone 분포
- Zone별 체류 분석
- Spot 상세 분석

### 5. DeepCon Forecast (예측)
- AI 기반 30분 후 리스크 예측
- Zone별 예측 정확도
- 시간대별 예측 성능

### 6. DeepCon Simulator (시뮬레이션)
- 실시간 리스크 히트맵
- 통계 vs Transformer 예측 비교
- Interactive 시간 슬라이더

---

## 💻 시스템 요구사항

### 최소 사양
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.9 이상
- **RAM**: 4GB 이상
- **Storage**: 10GB 이상 (데이터 포함)

### 권장 사양
- **RAM**: 8GB 이상
- **CPU**: 4 cores 이상
- **Storage**: SSD 권장

---

## 🚀 설치 방법

### 1. 레포지토리 클론

```bash
git clone <repository-url>
cd DeepCon
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
# - APP_PASSWORD: 대시보드 비밀번호
# - APP_ENV: production/development
# - 기타 설정
```

### 5. 데이터 준비

```bash
# 데이터 파일을 Datafile/ 디렉토리에 배치
# 전처리 실행
python src/precompute_optimized.py
python src/precompute_simulator.py
python src/precompute_forecast.py
```

---

## 🎮 사용 방법

### 로컬 실행

```bash
# 직접 실행
streamlit run main.py

# 백그라운드 실행
./start.sh

# 중지
./stop.sh
```

브라우저에서 `http://localhost:8501` 접속

### Docker 실행

```bash
# Docker Compose 사용
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

---

## 🚢 배포

### 프로덕션 배포

자세한 배포 가이드는 [DEPLOYMENT.md](DEPLOYMENT.md) 참조

```bash
# 1. 환경 설정
cp .env.example .env
# .env 편집

# 2. Docker 빌드
docker-compose build

# 3. 실행
docker-compose up -d

# 4. 헬스체크
curl http://localhost:8501/_stcore/health
```

### 배포 체크리스트

배포 전 [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) 확인

---

## 🔧 문제 해결

### 일반적인 문제

#### 1. 포트 충돌 (8501)

```bash
# 사용 중인 프로세스 확인
lsof -i :8501

# 프로세스 종료
kill -9 <PID>
```

#### 2. 모듈 import 에러

```bash
# 가상환경 활성화 확인
which python

# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

#### 3. 캐시 오류

```bash
# 캐시 클리어
rm -rf Cache/*
rm -rf __pycache__
rm -rf src/__pycache__
```

#### 4. 메모리 부족

```bash
# Docker 메모리 증가
docker run --memory="8g" ...

# 또는 docker-compose.yml에서 설정
```

### 로그 확인

```bash
# 애플리케이션 로그
tail -f logs/deepcon_$(date +%Y%m%d).log

# Streamlit 로그
tail -f logs/streamlit_*.log

# Docker 로그
docker-compose logs -f
```

---

## 📚 문서

- [배포 가이드](DEPLOYMENT.md)
- [배포 체크리스트](DEPLOYMENT_CHECKLIST.md)
- [릴리스 노트](RELEASE_NOTES.md)
- [함수 카탈로그](FUNCTION_CATALOG.md)

---

## 🏗️ 아키텍처

```
DeepCon/
├── main.py                  # 메인 애플리케이션
├── main_backup.py           # 레거시 통합 버전
├── src/
│   ├── cached_data_loader.py   # 데이터 로더
│   ├── config.py               # 설정
│   ├── env_config.py           # 환경 변수 관리
│   ├── logging_config.py       # 로깅 설정
│   ├── error_handlers.py       # 에러 핸들링
│   ├── tabs/                   # 탭 모듈
│   ├── components/             # UI 컴포넌트
│   └── utils/                  # 유틸리티
├── Cache/                   # 캐시 파일
├── Datafile/                # 원본 데이터
├── logs/                    # 로그 파일
└── .env                     # 환경 변수
```

---

## 🔒 보안

- **비밀번호 보호**: 환경 변수로 관리
- **HTTPS**: 프로덕션 환경 권장
- **방화벽**: 필요한 포트만 개방
- **로깅**: 민감 정보 제외

---

## 📈 성능

- **캐싱**: Streamlit `@st.cache_data` 활용
- **최적화**: Parquet 파일 형식
- **병렬 처리**: Multi-worker 지원
- **메모리**: 효율적인 데이터 처리

---

## 🤝 기여

현재는 내부 프로젝트로 외부 기여를 받지 않습니다.

---

## 📞 지원

- **이슈**: GitHub Issues
- **이메일**: support@tjlabs.com
- **문서**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📄 라이선스

Proprietary - SK Ecoplant / TJLABS

---

## 👥 개발팀

**TJLABS**  
SK Ecoplant 협력

---

## 📅 업데이트

마지막 업데이트: 2026-01-02  
버전: 2.0.0

---

**Made with ❤️ by TJLABS**
