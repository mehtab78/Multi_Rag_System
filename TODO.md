# TODO

## Before Publishing

- [x] Add screenshots or a short demo GIF to the README. Added `docs/demo.svg` preview.
- [x] Add a small test suite for retrieval scoring and prompt formatting. Added `tests/`.
- [x] Consider adding a sample `.env` validation command. Added `python3 check_setup.py`.
- [x] Add license information. Added MIT `LICENSE`.

## Future Improvements

- [x] Add uploaded-PDF support from the Gradio UI.
- [x] Store page numbers and chunk IDs as retrievable source metadata.
- [x] Add reranking for the top retrieved chunks.
- [x] Add evaluation examples with expected source chunks.

## Next Polish

- [x] Replace `docs/demo.svg` with a real screenshot or GIF after running the app with indexed data. Added `docs/app-screenshot.png` from the running Gradio UI.
- [x] Add an automated evaluation script that consumes `eval_examples.json`. Added `evaluate_examples.py`.
- [x] Add page-level citation rendering in the Gradio answer panel. Answers now append a page/chunk-level `Sources` section.

## Remaining Manual Polish

- [ ] Recapture `docs/app-screenshot.png` after setting a real `GEMINI_API_KEY`, starting OpenSearch/Ollama, ingesting PDFs, and generating a populated answer.
- [ ] Run `python3 evaluate_examples.py --generate` after the index is populated and review `eval_results.json`.
