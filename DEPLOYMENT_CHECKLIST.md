# DeepCon 배포 체크리스트

## ✅ 배포 전 최종 확인사항

### 1. 보안 설정
- [ ] `.env` 파일에 강력한 비밀번호 설정 (`APP_PASSWORD`)
- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] 프로덕션 환경에서 기본 비밀번호 사용 안 함
- [ ] 민감한 정보가 코드에 하드코딩되지 않았는지 확인

### 2. 데이터 준비
- [ ] `Datafile/` 디렉토리에 필요한 데이터 파일 존재
- [ ] `Cache/` 디렉토리 생성 및 권한 확인
- [ ] 전처리 스크립트 실행 완료
  - [ ] `python src/precompute_optimized.py`
  - [ ] `python src/precompute_simulator.py`
  - [ ] `python src/precompute_forecast.py`

### 3. 의존성 확인
- [ ] Python 3.9 이상 설치
- [ ] `requirements.txt` 패키지 모두 설치
  ```bash
  pip install -r requirements.txt
  ```
- [ ] PyTorch CUDA 설정 (GPU 사용 시)

### 4. 설정 파일
- [ ] `.env.example`을 `.env`로 복사
- [ ] `.env` 파일의 모든 필수 값 설정
- [ ] `.streamlit/config.toml` 프로덕션 설정 확인
- [ ] `APP_ENV=production` 설정

### 5. 성능 최적화
- [ ] 캐시 TTL 설정 확인 (`CACHE_TTL`)
- [ ] 워커 수 설정 (`MAX_WORKERS`)
- [ ] 메모리 사용량 모니터링 준비

### 6. 로깅
- [ ] `logs/` 디렉토리 생성 및 권한 확인
- [ ] 로그 레벨 설정 (`LOG_LEVEL=INFO`)
- [ ] 로그 파일 로테이션 설정 (선택)

### 7. 테스트
- [ ] 로컬 환경에서 정상 작동 확인
  ```bash
  streamlit run main.py
  ```
- [ ] 모든 탭 기능 테스트
- [ ] 에러 핸들링 테스트
- [ ] 로그 출력 확인

### 8. Docker (선택)
- [ ] Dockerfile 빌드 테스트
  ```bash
  docker build -t deepcon:latest .
  ```
- [ ] Docker 컨테이너 실행 테스트
  ```bash
  docker-compose up -d
  ```
- [ ] 컨테이너 헬스체크 확인

### 9. 네트워크 설정
- [ ] 방화벽 규칙 설정 (8501 포트)
- [ ] 리버스 프록시 설정 (Nginx 등)
- [ ] SSL/TLS 인증서 설정 (HTTPS)
- [ ] 도메인 DNS 설정

### 10. 모니터링
- [ ] 애플리케이션 로그 모니터링 설정
- [ ] 시스템 리소스 모니터링 (CPU, 메모리, 디스크)
- [ ] 에러 알림 설정
- [ ] 백업 전략 수립

---

## 🚀 배포 명령어 (프로덕션)

### 직접 실행
```bash
# 백그라운드 실행
nohup streamlit run main.py --server.port 8501 > logs/streamlit.log 2>&1 &

# 프로세스 확인
ps aux | grep streamlit

# 중지
pkill -f "streamlit run main.py"
```

### Docker 실행
```bash
# 빌드 및 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 재시작
docker-compose restart

# 중지
docker-compose down
```

---

## 📊 배포 후 확인사항

### 1. 즉시 확인 (1시간 이내)
- [ ] 애플리케이션 접속 가능 여부
- [ ] 로그인 기능 정상 작동
- [ ] 모든 탭 로딩 확인
- [ ] 데이터 표시 정상 확인
- [ ] 로그 파일 생성 확인

### 2. 단기 모니터링 (1일)
- [ ] 메모리 누수 없음
- [ ] CPU 사용률 정상 범위
- [ ] 응답 시간 모니터링
- [ ] 에러 로그 확인
- [ ] 사용자 피드백 수집

### 3. 장기 모니터링 (1주일+)
- [ ] 장시간 안정성 확인
- [ ] 성능 저하 없음
- [ ] 로그 파일 크기 관리
- [ ] 백업 정상 작동
- [ ] 정기 업데이트 계획

---

## 🔄 롤백 계획

문제 발생 시:
1. 즉시 이전 버전으로 롤백
2. 로그 파일 백업
3. 문제 원인 분석
4. 수정 후 재배포

```bash
# Docker 롤백
docker-compose down
docker tag deepcon:latest deepcon:backup
# 이전 이미지로 복구
docker-compose up -d

# 직접 실행 롤백
git checkout <previous-commit>
pkill -f "streamlit run main.py"
streamlit run main.py &
```

---

**체크리스트 작성일**: 2026-01-02
**작성자**: TJLABS
**버전**: 2.0.0
