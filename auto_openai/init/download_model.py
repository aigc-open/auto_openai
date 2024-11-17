import tiktoken, time

start_time = time.time()
tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
print(f"加载titoken: {time.time() - start_time}")