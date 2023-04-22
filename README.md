# chatpdf
A langchain, vectordatabase approach to talking with pdf on the internet

<img width="969" alt="Screenshot 2023-04-22 at 1 36 49 AM" src="https://user-images.githubusercontent.com/15062620/233744521-f38c475b-37a6-43d8-b258-30a238f8c0be.png">


## usage

`docker build -t chatpdf`

Create `secrets.toml` with `echo 'password = "password"' > secrets.toml`

`docker run --rm -e OPENAI_API_KEY=<key> -v `pwd`/secrets.toml:/app/.streamlit/secrets.toml -p 8501:8501 chatpdf`

Add `--gpus all` to enable the nvidia-runtime (leverage your GPU).
