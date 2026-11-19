# Adaptive Memory (for Open WebUI)

A long-term memory filter for [Open WebUI](https://github.com/open-webui/open-webui). It reads your conversations in the background, saves lasting facts in Open WebUI's own memory vault, and puts the relevant ones in front of the model before each reply. It was built and tuned for **AI companion personas**: one copy of the function per persona, each with its own settings.

The code in this repository was **written by Claude (Anthropic)** in conversation with DarkwingDuck13, who designed, tested and ran it. See [NOTICE.md](NOTICE.md) for credits and the lineage it grew from.

> **Status:** used daily on one setup (Open WebUI 0.13.0, Ollama). Changes were checked with stub tests and live use, not a full test suite. Expect rough edges and read the valves before you enable things.

## What it does

- **Extracts and stores facts** from your messages with a language model, tags them, scores them, and removes near-duplicates and contradictions.
- **Injects the relevant memories** into a short block before your message each turn. The block is stripped from earlier turns so history does not fill up with copies.
- **Knows what time it is.** An optional line gives the current weekday, date and time (your time zone), so the model stops guessing the date.
- **Keeps personas separate where it matters.** Each copy has a name, and a persona's own notes and reflections (lines that start with `Name:`) can be hidden from other personas on the same account. Facts about the user are shared by default.
- **Optional private writing:** self-notes with `[remember: ...]`, periodic reflections, and diary offers. All off or opt-in by default.
- **Save-time quality checks:** a proposed memory is rejected, and only the reason is logged, if it is a raw quote with a speaker label, contains `*actions*`, starts with "You" or "I", says "you" or "your" (ambiguous when several personas share facts), describes the current scene, or leaks an example word from the prompt.
- **Source stamp:** each new memory records which persona's conversation it came from (`[Source: Name]`), so memories can later be kept separate, shared, or shared on request. Nothing reads it yet.
- **Ollama-friendly background calls:** `think: false` is sent on the filter's own requests to Ollama's native `/api/chat`, which cuts hidden reasoning time sharply on models that think by default.
- **Status messages** in the chat (🧠) for recall and save, plus an optional pop-up notice fallback.
- **Privacy in logs:** the logging is designed to record counts, reasons and ids, not memory text (check your own logs before sharing them).

Compared with the lineage it builds on, the Mem0 mirror and the Prometheus metrics were removed, and the legacy "no effect" valves were dropped.

## Install

1. Open WebUI, Admin Panel, Functions, add a new function, paste `adaptive_memory.py`, save and enable it. Attach it to your model(s) as a filter.
2. **One copy per persona.** Valves are per function, so give each persona her own copy, set her `assistant_name`, and (for several personas on one account) set `persona_namespace` and `hide_notes_from_personas`.
3. Update **all copies at the same time** when you upgrade. The memory format gained a `[Source: ...]` field, and an older copy does not parse it.
4. Needs `pytz` (normally present in Open WebUI).

## First-time settings (the valves are grouped and tagged)

Valves appear in this order, with a tag at the start of each description: **Models**, **Persona**, **Display**, **Her writing**, **Save**, **Duplicates**, **Retrieval**, **Core notes**, **Background**, **Logging**.

At minimum set: `assistant_name`, `user_display_name`, the `llm_*` connection (provider, model and endpoint), `light_model_name`, and the two time zone valves (`timezone`, `current_time_timezone`).

## Recommended setup (Ollama)

- **Use a separate small model for background jobs** (`light_model_name`). If your chat model uses `OLLAMA_NUM_PARALLEL=1`, every background call otherwise queues behind your chat and can evict its cached context. A small CPU-only copy keeps background work out of the chat's way. Create one with a Modelfile that sets `PARAMETER num_gpu 0` and `PARAMETER num_ctx 8192`, then check `ollama ps`: the chat model should show GPU and the small one CPU.
- Keep `ollama_think_off` on (default) unless your endpoint rejects the field.
- The filter's own requests send no `num_ctx` or `keep_alive`. Set `OLLAMA_KEEP_ALIVE` on the server and `PARAMETER num_ctx` in your Modelfiles so models do not reload at the wrong size.

## Tools

`tools/memory_review.py` is a **read-only** health report for your memory vault: counts, flagged lines (raw quotes, speaker labels, "you" in shared facts, near-duplicates), and an export file of facts about the user. It never prints or exports a persona's own writing. Delete memories through the Open WebUI memory page, not SQL, so the vector index stays in step.

## Known issues

- In Open WebUI 0.13.0 the status line above replies did not render for us even though the events were sent and saved. The `status_toast` valve shows them as pop-ups instead.
- Not tested against other providers beyond what the original supported.
- Memory extraction quality depends heavily on the model you give it. Small models make attribution mistakes ("who said what"); the save-time checks catch some but not all.

## Related

The [Impetus](https://github.com/DarkwingDuck13/Impetus) drive system (separate project) can read this filter's contact file and write private notes into the memory vault through the `impetus_*` valves. Neither project needs the other.

## License

MIT, see [LICENSE](LICENSE). Credits and provenance are in [NOTICE.md](NOTICE.md).
