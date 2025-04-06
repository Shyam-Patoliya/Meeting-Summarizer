from typing import Optional
import fire
from models.datatypes import RawMessage, StopReason
from models.llama3.generation import Llama

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    world_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
    )

    dialogs = [
        [RawMessage(role="user", content="what is the recipe of mayonnaise?")],
        [
            RawMessage(
                role="user",
                content="I am going to Paris, what should I see?",
            ),
            RawMessage(
                role="assistant",
                content=,
                stop_reason=StopReason.end_of_turn,
            ),
            RawMessage(role="user", content="What is so great about #1?"),
        ],
        [
            RawMessage(role="system", content="Always answer with Haiku"),
            RawMessage(role="user", content="I am going to Paris, what should I see?"),
        ],
        [
            RawMessage(role="system", content="Always answer with emojis"),
            RawMessage(role="user", content="How to go from Beijing to NY?"),
        ],
    ]
    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
