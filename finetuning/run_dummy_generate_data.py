import os
from pathlib import Path
from generate_data import generate_cycle, _next_cycle_id
from dotenv import load_dotenv
import model_2b
import model_8b

load_dotenv("../../.env")

def main():
    base_dir = os.path.dirname(__file__)
    os.chdir(base_dir)

    prompts = [
        {
            "persona_id": 101,
            "product_id": 1001,
            "send_purpose": "repurchase",
            "has_event": 0,
            "event_content": "",
        },
        {
            "persona_id": 202,
            "product_id": 2002,
            "send_purpose": "cross_sell",
            "has_event": 1,
            "event_content": "limited time offer",
        },
    ]

    cycle_id = _next_cycle_id("finetuning_data")
    generate_cycle(cycle_id, model_2b, model_8b, prompts)
    print(f"Saved: finetuning_data/cycle_{cycle_id:02d}.csv")


if __name__ == "__main__":
    main()
