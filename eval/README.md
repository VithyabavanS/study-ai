# Evaluation set

Documents (download into `data/eval_docs/`):
- `attention.pdf`: Vaswani et al., "Attention Is All You Need", https://arxiv.org/abs/1706.03762

- `photosynthesis_ta.pdf`: Tamil Wikipedia, "ஒளித்தொகுப்பு", https://ta.wikipedia.org/wiki/ஒளித்தொகுப்பு
  (text under CC BY-SA 4.0). Questions: `questions_ta.jsonl`.

`questions.jsonl`: one test question per line. `evidence` is an exact phrase
from the document that supports the answer; `null` means the document cannot
answer the question.

