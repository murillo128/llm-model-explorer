# Routed-expert compact graph evidence for issue #237

The installed checkpoints were the pinned DeepSeek-V2-Lite
`9fc357346aeba67950a34a86f3520fc276ca2daf`, GLM-4.7-Flash
`25624b53414e585bcf7dcb9584667c3106c6089b`, and Kimi Linear
`5d029d1844aa64ec302e14466be7d0353c6e697f` references from issue #178.
The local model files stayed read-only and outside Git. These measurements used
their actual configuration and physical/numeric inventories, but did not read
weight values. Bytes below are compact UTF-8 JSON graph bytes, excluding the
small HTTP response envelope. The decoded response limit remains 33,554,432 B.

| Reference | Full graph before (B) | Compact graph after (B) | Nodes before → after | Edges before → after | Parameters | Expert instances / definitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek | 36,836,878 | 27,530,058 | 7,955 → 2,963 | 20,627 → 13,971 | 5,291 | 1,664 / 26 |
| GLM | 35,170,355 | 26,817,109 | 5,430 → 2,486 | 19,082 → 19,082 | 9,491 | 2,944 / 46 |
| Kimi | 30,925,226 | 30,925,226 | 7,869 → 7,869 | 2,351 → 2,351 | 20,493 | 6,656 / 0 on the actual response |

The before pass used the same builders with a temporary 128 MiB **measurement
only** construction cap so complete records could be counted. It did not change
the served 32 MiB limit. Node/edge/parameter contributions before compaction
were respectively 14,655,385 / 7,978,635 / 13,273,089 B for DeepSeek and
14,429,727 / 7,339,941 / 11,812,468 B for GLM. The limit was exceeded by
3,282,446 B and 1,615,923 B respectively. The remaining bytes were symbols,
repetitions, verified presentation templates, diagnostics and JSON framing.

After compaction, those three contributions were 6,243,605 / 5,426,059 /
13,273,089 B for DeepSeek and 4,428,711 / 7,339,941 / 11,812,468 B for GLM.
The compact family records added 1,669,135 B and 1,650,645 B respectively.
All concrete parameter records and router connections remain in the response.
The backend reconstructed and validated each compact graph against the exact
admitted inventory before the after measurement. A deterministic Kimi fixture
also exercised this representation for all 6,656 routed experts: its graph went
from 25,328,941 B to 19,289,659 B using 26 family definitions. Kimi's real
graph remains in its original wire form because it already fits the limit.

The production HTTP service, started with the pinned local references, served
`available` architecture responses of 27,530,166 B for DeepSeek and 26,817,223 B
for GLM, including their response envelopes. Both responses passed the
independent API architecture conformance check against the corresponding
session inventory. The served graphs retained all 1,664 DeepSeek and 2,944 GLM
expert instances.

The measurements establish static graph size and metadata closure. They do not
establish numerical inference, prompt-dependent routing or weight decoding.
