import os
import requests

# 1. Forge the 16k Context Model
print("🔨 Building qwen-director with 16k context...")
os.system("echo 'FROM qwen2.5vl' > Modelfile")
os.system("echo 'PARAMETER num_ctx 16384' >> Modelfile")
os.system("ollama create qwen-director -f Modelfile")

# 2. Verify the model's DNA
print("\n🔍 Inspecting the new model's parameters...")
response = requests.post("hhttps://d467-35-247-145-32.ngrok-free.app/v1", json={"name": "qwen-director"})

if response.status_code == 200:
    data = response.json()
    parameters = data.get("parameters", "").strip()
    print("\n" + "="*40)
    print("ACTIVE ENFORCED PARAMETERS:")
    print("="*40)
    print(parameters if parameters else "⚠️ Failed to apply parameters.")
    print("="*40)
else:
    print(f"❌ Verification failed. Status code: {response.status_code}")