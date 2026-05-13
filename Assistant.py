import ollama
import os

# Task Classifier
def classify(prompt, image_path=None):
    prompt_lower = prompt.lower()

    if image_path:
        return "vision"

    coder_keywords = [
        "code", "python", "c++", "java", "bug", "debug",
        "leetcode", "algorithm", "function", "implement",
        "class", "script", "error", "traceback", "dfs", "bfs"
    ]

    if any(word in prompt_lower for word in coder_keywords):
        return "coder"

    return "chat"


# Extract image path from input
# Format example:
# "what is in this image: /home/user/pic.jpg"
def extract_image(text):
    keywords = [".png", ".jpg", ".jpeg", ".webp"]

    for word in text.split():
        if any(word.endswith(ext) for ext in keywords) and os.path.exists(word):
            return word

    return None


# Main loop
def main():
    print("🚀 Unified Ollama Agent Started")
    print("Type 'exit' to quit\n")

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            print("Bye 👋")
            break

        # detect image path inside text
        image_path = extract_image(user)

        task = classify(user, image_path)

        # Model selection
        if task == "coder":
            model = "qwen2.5-coder:3b"

        elif task == "vision":
            model = "llava:latest"

        else:
            model = "phi4-mini:latest"


        # Build request
        try:
            if task == "vision":
                # remove image path from text for cleaner prompt
                cleaned_prompt = user.replace(image_path, "").strip()

                response = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": cleaned_prompt or "What is in this image?",
                        "images": [image_path]
                    }]
                )
            else:
                response = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": user
                    }]
                )

            print(f"\n[{model}]")
            print(response["message"]["content"])
            print()

        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()


"""
🧠 Chat
You: explain neural networks
💻 Coding
You: write python code for dfs
🖼️ Image (your idea)
You: what is in this image /home/priyanshu/pic.jpg

or even:

You: describe this /home/priyanshu/Desktop/dog.png
"""