# file_name 삭제
df_processed.drop(columns=["file_name"], inplace=True)

# JSON 저장
df_processed.to_json(
    "insta_laneige_json_processing.json",
    orient="records",
    force_ascii=False,
    indent=2
)