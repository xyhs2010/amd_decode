1. 可以自定义 _layer,并将自定义的 _layer 和原实现做 byte 级比对
2. 把 kvcache 放在 host 内存，并和原实现对齐
3. 原实现跑分(跑起来太慢了，得先加速)
4. 自己写 kernel 来实现