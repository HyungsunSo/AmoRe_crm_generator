# CRM Message Studio 사용 가이드

본 프로젝트의 프론트엔드는 JSON 데이터를 동적으로 불러오기 때문에, 보안 정책상 **로컬 서버**를 통해 실행해야 합니다.

## 1. 실행 방법 (로컬 서버 띄우기)

터미널에서 프로젝트 루트 디렉토리(`/AmoRe_crm_generator`)로 이동한 뒤 아래 명령어를 입력하세요.

```bash
# Python 3가 설치되어 있는 경우
python3 -m http.server 8888
```

## 2. 접속 주소

서버가 실행되면 브라우저에서 아래 주소로 접속하세요.

- **기본 접속**: [http://localhost:8888/frontend/toss-v2.html](http://localhost:8888/frontend/toss-v2.html)
- **메인 인덱스**: [http://localhost:8888/frontend/index.html](http://localhost:8888/frontend/index.html)

## 3. 주의 사항

- HTML 파일을 직접 더블 클릭해서 열 경우 (`file://...`), 브라우저의 **CORS 보안 정책**으로 인해 제품 및 브랜드 데이터를 불러오지 못할 수 있습니다.
- 반드시 `http://localhost:8888` 서버를 통해 접속해야 모든 기능이 정상 작동합니다.
- 만약 8888 포트가 사용 중이라면 다른 포트 번호(예: 8000)를 사용해도 됩니다.
