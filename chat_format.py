"""chat_format.py — the conversation template shared by fine-tuning and chat.

A continuation model only learns to chat if we train and prompt it with a
consistent turn structure. We use a small, BPE-friendly template and the GPT-2
end-of-text token (id 50256) as the end-of-turn signal, so the model has a clean
token to emit when it's done — and the chat loop has a clean token to stop on.

    <prompt>
    User: hi
    Assistant: hello!<eot>
    User: how are you?
    Assistant: great, thanks!<eot>

`encode_example` returns (token_ids, loss_mask): the mask is 0 on the prompt /
user tokens and 1 on the assistant's reply (incl. the trailing <eot>), so the
model is only trained to *produce* assistant turns, not to parrot the user.
"""
import tiktoken

ENC = tiktoken.get_encoding('gpt2')
EOT = ENC.eot_token                      # 50256 — <|endoftext|>, our end-of-turn

SYSTEM_PREFIX = (
    "The following is a conversation with a helpful, friendly AI assistant.\n"
)
USER_TAG = "User: "
ASSISTANT_TAG = "Assistant: "


def _enc(s):
    return ENC.encode_ordinary(s)


def render_prompt(history, system=SYSTEM_PREFIX):
    """Render conversation history into the text the model should continue from,
    ending right after `Assistant: ` so the next token begins the reply.

    history: list of (role, content) with role in {"user", "assistant"}.
    """
    parts = [system] if system else []
    for role, content in history:
        tag = USER_TAG if role == 'user' else ASSISTANT_TAG
        parts.append(f"{tag}{content}\n")
    parts.append(ASSISTANT_TAG)
    return "".join(parts)


def encode_prompt(history, system=SYSTEM_PREFIX):
    """Token ids for the prompt the model continues (no trailing reply)."""
    return _enc(render_prompt(history, system))


def encode_example(messages, system=SYSTEM_PREFIX):
    """Encode a full conversation into (ids, loss_mask) for supervised fine-tuning.

    messages: list of {"role": "user"|"assistant", "content": str}.
    Loss is masked (0) everywhere except assistant replies + their <eot>.
    """
    ids, mask = [], []

    def add(text, train):
        toks = _enc(text)
        ids.extend(toks)
        mask.extend([1 if train else 0] * len(toks))

    if system:
        add(system, train=False)
    for m in messages:
        if m['role'] == 'user':
            add(f"{USER_TAG}{m['content']}\n", train=False)
        else:
            add(ASSISTANT_TAG, train=False)            # the tag is a cue, not a target
            add(f"{m['content']}\n", train=True)        # learn the reply...
            ids.append(EOT); mask.append(1)             # ...and to stop with <eot>
    return ids, mask


# Tokens that, if generated, mean the assistant turn is over and we should stop.
# EOT is the clean signal; "\nUser:" is a fallback for an undertrained model that
# rambles into a fake user turn.
STOP_TEXT = "\n" + USER_TAG.rstrip()
