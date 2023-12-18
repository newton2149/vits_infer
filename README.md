# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## For Setup

#### Step 1 (Clone the git repo)

```sh

git clone https://github.com/newton2149/vits_kinyarwanda.git

```

#### Step 2 (Install Required Dependencies) 
```sh

pip3 install -r requiremnts.txt

#Install ESpeak Engine
apt-get install espeak

mkdir models

#Add Models to the models directory
```


#### Step 3 (Inference Code)
```sh
python3 websocket-server.py
# open another terminal and run
python3 test.py

```
##### You Can obtain either the audio file or a audio_files.zip if you send a text file
