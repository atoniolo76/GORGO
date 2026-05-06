# Network Traffic Characteristics of Data Centers in the Wild

- **Bibkey**: `benson2010imc`
- **Authors**: Theophilus Benson, Aditya Akella, David A. Maltz
- **Venue / Year**: ACM Internet Measurement Conference (IMC), 2010
- **arXiv / DOI**: 10.1145/1879141.1879175
- **URL**: https://conferences.sigcomm.org/imc/2010/papers/p267.pdf

## One-paragraph summary
A measurement study of traffic across 10 data centers (university, enterprise, and cloud) that characterizes flow size, flow inter-arrival times, link utilization, and temporal patterns. Among other findings, the paper documents pronounced diurnal and weekday/weekend variation in aggregate utilization and shows that traffic patterns differ substantially across data-center types. It is a standard citation for "diurnal demand" claims in distributed-systems papers that motivate cross-region or cross-cluster capacity provisioning.

## Supports our claim
> "Operators distribute model replicas across multiple geographies for three reasons: to follow diurnally shifting demand from a globally distributed user base, ..." (lines 138–141)

The diurnal-demand claim currently relies on `\citep{skywalker,skyserve}`. Skywalker does cover regional diurnal patterns (verified by abstract: "SkyWalker aggregates regional diurnal patterns through cross-region traffic handling"), so the existing citations are defensible. `benson2010imc` is offered as an *additional* canonical empirical reference if the author wants a measurement-paper anchor that predates the LLM-serving literature. **Uncertain — the existing cites already suffice; mark this as optional.**

## Suggested citation site
§Introduction, around line 142: optionally change `\citep{skywalker,skyserve}` to `\citep{skywalker,skyserve,benson2010imc}` if the author wants an empirical-measurement anchor for the diurnal-demand claim. Skip if not.

## BibTeX
```
@inproceedings{benson2010imc,
  title     = {Network Traffic Characteristics of Data Centers in the Wild},
  author    = {Benson, Theophilus and Akella, Aditya and Maltz, David A.},
  booktitle = {Proceedings of the 10th ACM SIGCOMM Conference on Internet Measurement (IMC)},
  pages     = {267--280},
  year      = {2010},
  doi       = {10.1145/1879141.1879175}
}
```
