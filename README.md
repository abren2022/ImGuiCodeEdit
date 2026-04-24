python translator.py 
['English', 'Chinese']
Chinese → English
Traceback (most recent call last):
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connection.py", line 198, in _new_conn
    sock = connection.create_connection(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/util/connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/socket.py", line 967, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno -3] Temporary failure in name resolution

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connectionpool.py", line 787, in urlopen
    response = self._make_request(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connectionpool.py", line 488, in _make_request
    raise new_e
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connectionpool.py", line 464, in _make_request
    self._validate_conn(conn)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connectionpool.py", line 1093, in _validate_conn
    conn.connect()
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connection.py", line 753, in connect
    self.sock = sock = self._new_conn()
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connection.py", line 205, in _new_conn
    raise NameResolutionError(self.host, self, e) from e
urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7a288e19ceb0>: Failed to resolve 'raw.githubusercontent.com' ([Errno -3] Temporary failure in name resolution)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/adapters.py", line 667, in send
    resp = conn.urlopen(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/connectionpool.py", line 841, in urlopen
    retries = retries.increment(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/urllib3/util/retry.py", line 519, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Max retries exceeded with url: /stanfordnlp/stanza-resources/main/resources_1.10.0.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7a288e19ceb0>: Failed to resolve 'raw.githubusercontent.com' ([Errno -3] Temporary failure in name resolution)"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/user/yaojie/Anomagic-main/demo/translator.py", line 9, in <module>
    aa = translation_zh_en.translate("朋友")
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/translate.py", line 64, in translate
    return self.hypotheses(input_text, num_hypotheses=1)[0].value
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/translate.py", line 328, in hypotheses
    translated_paragraph = self.underlying.hypotheses(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/translate.py", line 201, in hypotheses
    apply_packaged_translation(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/translate.py", line 451, in apply_packaged_translation
    sentences = sentencizer.split_sentences(input_text)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/sbd.py", line 160, in split_sentences
    doc = self.lazy_pipeline()(text)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/argostranslate/sbd.py", line 149, in lazy_pipeline
    self.stanza_pipeline = stanza.Pipeline(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/stanza/pipeline/core.py", line 208, in __init__
    download_resources_json(self.dir,
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/stanza/resources/common.py", line 459, in download_resources_json
    request_file(
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/stanza/resources/common.py", line 157, in request_file
    download_file(url, temppath, proxies, raise_for_status)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/stanza/resources/common.py", line 119, in download_file
    r = requests.get(url, stream=True, proxies=proxies)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/api.py", line 73, in get
    return request("get", url, params=params, **kwargs)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
  File "/home/user/miniforge3/envs/vllm/lib/python3.10/site-packages/requests/adapters.py", line 700, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='raw.githubusercontent.com', port=443): Max retries exceeded with url: /stanfordnlp/stanza-resources/main/resources_1.10.0.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7a288e19ceb0>: Failed to resolve 'raw.githubusercontent.com' ([Errno -3] Temporary failure in name resolution)"))
