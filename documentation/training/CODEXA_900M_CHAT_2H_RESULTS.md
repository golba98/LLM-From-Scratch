# Codexa 934M two-hour experimental results

COMPLETED safely at the 4,000 optimizer-step cap, without using the full two-hour ceiling. Total session time: 6,326.50 seconds (1h45m26.5s). No training process remains.

Processed 32,000 conversations (17.50% of the 182,892-record pool), 25,912,988 supervised assistant tokens and 33,465,911 context tokens. Sources: OASST1 4,715; UltraChat 27,285.

Common validation loss fell from 2.229543 to 1.964267. The 64-case deterministic development checks do not establish conversational improvement:

| Metric | Initialization | Final step4000 |
|---|---:|---:|
| Literal task macro accuracy |14.29%|10.71%|
| Model-generated END |68.75%|62.50%|
| Severe degeneration |35.94%|40.63%|

Intermediate task macros ranged10.71–12.50%; none surpassed initialization. Keyword/literal diagnostics have limited semantic coverage. Protected acceptance was not run, as configured for this experiment. The run's `best` pointer remains step0; it is a quality-qualified selection, not lowest validation loss. No new model was promoted or historical artifact deleted.

Final full recovery checkpoint: `outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints/slot-2/checkpoint.pt`, step4000, 5,636,070,993 bytes. SHA256 `374f3f1746b0ff0b24febc5176f3d20f5bb4ccadc3fb6aa24a120d1f31481918`. Post-run hash and actual CPU deserialization verified; native model/optimizer/scheduler/RNG and sampler metadata remain available. This inspection does not claim a new continuation update was tested.

Recommendation: preserve the full recovery state for any authorized further session, but investigate repetition/termination and sampling before simply repeating this adaptation schedule. Additional training time is not authorized by this completion check. A new session needs a new budget, with explicit checkpoint and sampler continuation semantics.
