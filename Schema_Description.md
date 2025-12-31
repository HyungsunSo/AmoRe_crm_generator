# Pipeline I/O Schema Specification

## Overview

본 문서는 페르소나 기반 추천·생성·선호 학습 파이프라인의  
각 단계별 입력(Input)과 출력(Output) 데이터 형식을 정의한다.

Pipeline Flow  
Stage 0 (Dataset)  
→ Stage 1 (Analyzer, Qwen/Qwen2.5-1.5B-Instruct)  
→ Stage 2 (Generator, LGAI-EXAONE/EXAONE-4.0-1.2B)  
→ Stage 3 (Discriminator, gpt-4o)

---

## Stage 0. Initial Dataset Input

### Description
학습 및 생성 파이프라인의 초기 입력 데이터셋이다.  
페르소나와 상품 간의 기본 관계를 정의한다.

### Input Format (CSV)

persona_id,product_id

### Field Definition

| Field | Type | Description |
|---|---|---|
| persona_id | int | 페르소나 식별자 |
| product_id | int | 추천 대상 상품 ID |
| send_purpose | string | 메시지 발신 목적 |
| has_event | bool(1, 0) | 이벤트 존재 여부, 랜덤 |
| event_content | string | 세부 이벤트 내용

---

## Stage 1. Analyzer

### Description
페르소나와 상품 정보를 해석하여  
Generator 단계에서 사용할 컨텍스트와 제약 조건을 생성한다.  
후보 생성 개수 n을 함께 입력받는다.

### Input Schema

- persona_id: int  
- product_id: int
- send_purpose: string
- has_event: bool(1, 0)
- event_content: string
- n: int

### Output Schema

- persona_id: int  
- product_id: int  
- analysis  
  - persona_summary: string  
  - product_summary: string  
  - key_preferences: list[string]  
  - key_constraints: list[string]  
- n: int  

---

## Stage 2. Generator

### Description
Analyzer 결과를 기반으로  
후보 응답 n개를 생성하고 각 응답에 태그 정보를 부착한다.

### Input Schema

- analysis  
  - persona_summary: string  
  - product_summary: string  
  - key_preferences: list[string]  
  - key_constraints: list[string]  
- n: int  

### Output Schema

- candidates: list  
  - response_id: int  
  - text: string 

---

## Stage 3. Discriminator

### Description
Generator가 생성한 후보 응답 n개를 평가하여  
점수와 순위를 산출한다.

### Input Schema

- analysis  
  - persona_summary: string  
  - product_summary: string  
  - key_preferences: list[string]  
  - key_constraints: list[string]  
- candidates: list  
  - response_id: int  
  - text: string 


### Output Schema

- analysis  
  - persona_summary: string  
  - product_summary: string  
  - key_preferences: list[string]  
  - key_constraints: list[string]  
- best  
  - response_id: int  
  - text: string  
  - tags: object  
- refined: list  
  - response_id: int  
  - text: string  
  - tags: object  

---

## Design Notes

- 모든 Stage는 stateless 구조로 설계 가능하다.  
- 각 단계의 출력은 JSON / JSONL / CSV 형태로 저장 가능하다.  
- Stage 단위 재실행 및 부분 재학습을 지원한다.  
- 실제 구현 시 Python dataclass 또는 Pydantic 모델로 매핑 가능하다.

---

## Document Naming Recommendation

- Pipeline_IO_Schema.md  
- Stagewise_Data_Contract.md  
