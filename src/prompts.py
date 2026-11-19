translation_prompt = (
    lambda dialect, src_text: f"Here are examples of {dialect}:\n"
    "Translate from Standard American English to {dialect} in a way that feels natural and preserves the original meaning and tone."
    "1. Only provide the translation. Do not mention or explain how the translation was done.\n"
    "2. Ensure the text sounds natural and realistic in {dialect}.\n\n"
    f"Translate the following text: '{src_text}'"
)

error_prompt = lambda question, answer: (
    "You are a fact checker.\nYou will be given a question-answer pair.\nYour task is to rewrite the answer while inserting exactly ONE subtle mistake. That mistake should make the answer worse.\nOnly provide the new answer. Do not mention or explain how the error was introduced. You should not modify any content apart from the error.\n\n"
    f"Your question is: '{question}'\n"
    f"Rewrite and introduce a mistake for the following answer: '{answer}'"
)
