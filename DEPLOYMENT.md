# DeepCon IRFM Dashboard - 배포 가이드

## 📋 배포 전 체크리스트

### 1. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (필수 항목)
# - APP_PASSWORD: 강력한 비밀번호 설정
# - APP_ENV: production으로 설정
# - LOG_LEVEL: 프로덕션 로그 레벨 (INFO 권장)
```

### 2. 의존성 확인

```bash
# Python 3.9+ 필요
python --version

# 패키지 설치
pip install -r requirements.txt
```

### 3. 데이터 준비

- `Datafile/` 디렉토리에 데이터 파일 배치
- `Cache/` 디렉토리 자동 생성됨

## 🚀 배포 방법

### 방법 1: 직접 실행 (개발/테스트)

```bash
# Streamlit 서버 실행
streamlit run main.py --server.port 8501

# 또는 백그라운드 실행
nohup streamlit run main.py --server.port 8501 > logs/streamlit.log 2>&1 &
```

### 방법 2: Docker 사용 (권장)

```bash
# Docker 이미지 빌드
docker build -t deepcon:latest .

# Docker 컨테이너 실행
docker run -d \
  --name deepcon \
  -p 8501:8501 \
  -v $(pwd)/Cache:/app/Cache \
  -v $(pwd)/Datafile:/app/Datafile \
  -v $(pwd)/logs:/app/logs \
  -e APP_PASSWORD=your_secure_password \
  deepcon:latest

# 또는 Docker Compose 사용
docker-compose up -d
```

### 방법 3: Docker Compose (가장 간편)

```bash
# 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

## 🔧 프로덕션 설정

### Nginx 리버스 프록시 (선택)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
    }
}
```

### SSL/TLS 설정 (HTTPS)

```bash
# Let's Encrypt 인증서 발급
sudo certbot --nginx -d your-domain.com
```

## 📊 모니터링

### 로그 확인

```bash
# 애플리케이션 로그
tail -f logs/deepcon_$(date +%Y%m%d).log

# Docker 로그
docker-compose logs -f deepcon
```

### 헬스 체크

```bash
# HTTP 헬스 체크
curl http://localhost:8501/_stcore/health

# Docker 헬스 상태
docker inspect --format='{{.State.Health.Status}}' deepcon
```

## 🔒 보안 고려사항

1. **비밀번호**: 반드시 환경변수로 강력한 비밀번호 설정
2. **방화벽**: 8501 포트 접근 제한 (필요한 IP만 허용)
3. **HTTPS**: 프로덕션에서는 반드시 HTTPS 사용
4. **로그**: 민감한 정보가 로그에 기록되지 않도록 주의
5. **업데이트**: 정기적인 보안 패치 적용

## 🔄 업데이트 절차

```bash
# 1. 코드 업데이트
git pull origin main

# 2. 의존성 업데이트
pip install -r requirements.txt --upgrade

# 3. 재시작
## 직접 실행 시
pkill -f "streamlit run main.py"
streamlit run main.py --server.port 8501 &

## Docker 사용 시
docker-compose down
docker-compose build
docker-compose up -d
```

## 📈 성능 최적화

1. **캐시 설정**: `.env`에서 `CACHE_TTL` 조정 (기본 3600초)
2. **워커 수**: `MAX_WORKERS` 설정으로 병렬 처리 조정
3. **메모리**: Docker의 경우 메모리 제한 설정
   ```bash
   docker run --memory="4g" --memory-swap="4g" ...
   ```

## 🐛 문제 해결

### 1. 포트 충돌

```bash
# 8501 포트를 사용 중인 프로세스 확인
lsof -i :8501

# 프로세스 종료
kill -9 <PID>
```

### 2. 캐시 문제

```bash
# 캐시 디렉토리 정리
rm -rf Cache/*
rm -rf __pycache__
rm -rf src/__pycache__
```

### 3. Docker 문제

```bash
# 컨테이너 재시작
docker-compose restart

# 완전히 재빌드
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 📞 지원

문제 발생 시:
1. 로그 파일 확인 (`logs/deepcon_*.log`)
2. GitHub Issues에 문제 보고
3. TJLABS 기술지원팀 연락

---

**마지막 업데이트**: 2026-01-02
**버전**: 2.0.0
**관리**: TJLABS
