1. 可以自定义 _layer,并将自定义的 _layer 和原实现做 byte 级比对
2. 把 kvcache 放在 host 内存，并和原实现对齐
3. 原实现跑分
4. 用 torch 实现简单的检索，并跑分