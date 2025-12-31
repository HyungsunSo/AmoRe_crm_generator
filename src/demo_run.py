#!/usr/bin/env python3
"""데모 러너: 페르소나 인덱스로 personas.json에서 페르소나 정보를 조회하여 실행

사용 예:
  python3 src/demo_run.py             # 기본값으로 실행 (Luxury_Lover, 설화수, 자음생크림)
  python3 src/demo_run.py --persona_idx 1 --brand SK-II  # 다른 페르소나/브랜드
  python3 src/demo_run.py --use_local_model  # 로컬 Qwen 모델로 생성
"""
import subprocess
import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(BASE, 'src', 'generate_marketing.py')
PERSONAS_JSON = os.path.join(BASE, 'data', 'personas.json')
DEMO_PERSONAS = os.path.join('data', 'demo_personas.json')
DEMO_PRODUCTS = os.path.join('data', 'demo_products.json')


def load_personas():
    """personas.json 로드"""
    with open(PERSONAS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_persona_name(persona_idx):
    """페르소나 인덱스로 페르소나 이름 조회"""
    personas = load_personas()
    if 0 <= persona_idx < len(personas):
        return personas[persona_idx].get('name', f'Persona_{persona_idx}')
    return f'Persona_{persona_idx}'


def build_cmd(use_real: bool, persona_idx: int, brand: str, product: str, top_k: int, use_local_model: bool = False, model_name: str = None):
    cmd = [sys.executable, GEN,
           '--persona', str(persona_idx),
           '--brand', brand,
           '--product', product,
           '--top_k', str(top_k)]
    if not use_real:
        cmd += ['--personas_path', DEMO_PERSONAS, '--products_path', DEMO_PRODUCTS]
    if use_local_model:
        cmd.append('--use_local_model')
        if model_name:
            cmd += ['--model_name', model_name]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_real', action='store_true', default=True, help='기본 데이터(`data/*.json`) 사용 (기본값: True)')
    parser.add_argument('--persona_idx', type=int, default=0, help='페르소나 인덱스 (기본값: 0 = Luxury_Lover)')
    parser.add_argument('--brand', default='설화수', help='브랜드명 (부분일치 허용, 기본값: 설화수)')
    parser.add_argument('--product', default='자음생크림 리치 단품세트 50ml', help='제품명(부분일치 허용, 기본값: 자음생크림 리치)')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K (기본값: 3)')
    parser.add_argument('--use_local_model', action='store_true', help='로컬 Qwen 모델로 마케팅 초안 생성')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B-Instruct', help='로컬 모델 ID')
    args = parser.parse_args()

    # 페르소나 인덱스로 페르소나 이름 조회
    persona_name = get_persona_name(args.persona_idx)
    print(f"[Demo Runner]")
    print(f"  페르소나: {args.persona_idx} ({persona_name})")
    print(f"  브랜드: {args.brand}")
    print(f"  제품: {args.product}")
    print(f"  Top-K: {args.top_k}")
    if args.use_local_model:
        print(f"  모델: {args.model_name} (로컬)")
    print()

    cmd = build_cmd(args.use_real, args.persona_idx, args.brand, args.product, args.top_k, 
                    args.use_local_model, args.model_name if args.use_local_model else None)
    subprocess.run(cmd, check=True)



if __name__ == '__main__':
    main()
