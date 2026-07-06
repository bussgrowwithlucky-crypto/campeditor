import requests
import base64

key = "sk-nry-tMee9Jo-n7gF1Y37dRHYQS0pRdJ3Kh9GtfHKjccBZPQ"
url = "https://router.bynara.id/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

# 1) Text test — mistral-large
text_payload = {
    "model": "mistral-large",
    "messages": [{"role": "user", "content": "ping. reply with just the word pong."}],
    "max_tokens": 16,
    "temperature": 0.0,
}

# 2) Vision test — mistral-medium-3-5 with a real 1x1 PNG
png_bytes = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
data_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

vision_payload = {
    "model": "mistral-medium-3-5",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
    "max_tokens": 64,
    "temperature": 0.0,
}

# 3) Vision test on the text model — confirm vision is text-model-disabled
text_model_vision_payload = {
    "model": "mistral-large",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
    "max_tokens": 64,
    "temperature": 0.0,
}

def call(label, payload):
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        print(f"--- {label} ---")
        print(f"status: {r.status_code}")
        print(f"body:   {r.text[:600]}")
        if r.status_code == 200:
            try:
                j = r.json()
                content = (
                    j.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                print(f"content: {content!r}")
            except Exception:
                pass
        print()
        return r.status_code
    except Exception as e:
        print(f"--- {label} ---")
        print(f"EXC: {e}\n")
        return None

s1 = call("TEXT — mistral-large", text_payload)
s2 = call("VISION — mistral-medium-3-5", vision_payload)
s3 = call("VISION attempt on TEXT model — mistral-large", text_model_vision_payload)

print("=== summary ===")
print(f"text(mistral-large):              {s1}")
print(f"vision(mistral-medium-3-5):       {s2}")
print(f"vision-attempt-on-text-model:     {s3}")
