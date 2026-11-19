# traceai-udah-scorer-poc
极简 Python 适配器，读取 traceAI 导出的 OTLP/GenAI 追踪数据，计算每轮的 U/D/A/H 对话一致性分数。针对 future-agi/traceAI#185 的范围限定 POC —— 仅使用标准库和合成数据，检测超越局部连贯性/解决率的跨轮语义漂移。
