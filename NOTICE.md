# Notice: credits and license provenance

## Who wrote what

- **Claude (Anthropic)** wrote the code for the changes made in this repository, working with DarkwingDuck13 in conversation. That includes the persona-aware features, the time line and status handling, the save-time checks, the source stamp, the Ollama request changes, the valve reorganization, the removal of the Mem0 and Prometheus code, and the memory review tool.
- **DarkwingDuck13** directed the design, tested every change on a live Open WebUI setup, and decided what was kept.

## Lineage (the code this builds on)

As far as we can establish, this started as the Adaptive Memory filter for Open WebUI by **AG**, published as "alexgrama" on openwebui.com (a version 3 post there), and was carried on in **crizCraig/adaptive_memory_owui** and in a fork by **donbcd** (donbcd/adaptive_memory_owui). We have not been able to confirm every step of that chain, and some of the history above is our best reading of the evidence, not something the authors have told us.

## License provenance

The two GitHub repositories above state in their READMEs that the project is MIT licensed, but neither contains a LICENSE file, and the files we received carry no copyright header, so there is no original copyright line to copy. We therefore release our changes under the standard MIT text in `LICENSE`, and we credit the earlier authors here.

If you are one of the earlier authors and want something changed (a copyright line, a different credit, or removal), please open an issue. We will fix it promptly.
