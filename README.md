# 👂 みみアラート

イヤホンで音楽を聴いていて周りの音が聞こえないときに、マイクで周囲の音を拾ってアラートを鳴らす Web アプリ。

- 話し声・インターホン・電話・アラーム・家電の音・泣き声などを検知したら、アラート音のあとに拾った音そのものを再生
- 判定は2段階：YAMNet（MediaPipe）で候補を拾い、AST（Transformers.js / WebGPU）で再判定
- 話し声を文字起こし（Web Speech API）

https://eatski.github.io/mimi-alert/
