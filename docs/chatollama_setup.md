# Running with ChatOllama

Disable OPENAI_API_KEY by setting the following two env variales:
(without both of these env vars ollama complains about auth)

```
$ export OPENAI_API_KEY=ollama
$ export OPENAI_BASE_URL=<YOUR OLLAMA ENDPOINT>
```

Example Auth Error

```
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: ollama. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
```
