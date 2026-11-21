# M5.1 — Exact replay after non-scientific SVG serialization fix

Original full world calculations for preregistered campaign [run #44913028853](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44913028853) completed, saved summary and per-world CSV, but a cosmetic SVG-serializer bug failed for extinct final populations. [Fix PR #30](https://github.com/kkongiu/neuroevolution-lab/pull/30) changes ONLY the representation of a list of survivor positions and adds an empty-population regression test; NO change to physics, seeds, action choices, resource or reproduction laws, scientific outcomes or protocol.

REPLAY the EXACT frozen workflow and parameters from [docs/M5_1_PROTOCOL.md](../docs/M5_1_PROTOCOL.md): main 73000–73015, shifted 74000–74007, 400 ticks, 4 active arms, 4 first-main-seed negative controls each, no retuning or post-hoc scoring. Publish both workflow runs and the original primary results without preference for the replay.
