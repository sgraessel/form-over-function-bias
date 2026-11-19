import json
import yaml

from model import get_model
from prompts import translation_prompt
from utils.utils import load_config, read_jsonl_file, write_file


def rewrite_answers(
    model_name_or_path: str = "gpt-4.1",
    src_file: str = "data/generated_answers/gpt-4.1-answers.json",
    target_language: str = "Indian English",
) -> None:

    config = load_config("config.yml")

    model = get_model(
        model_name_or_path=model_name_or_path,
        config=config,
    )

    original_dicts = read_jsonl_file(src_file)
    new_dicts = [dict(dictionary) for dictionary in original_dicts]

    texts = [
        [entry["answers"]["answer1"]["answer"] for entry in original_dicts],
        # [entry["answers"]["answer2"]["answer"] for entry in original_dicts],
    ]

    prompt = translation_prompt
    file_ending = target_language + ".json"

    prompts = [prompt(target_language, answer) for answer in texts[1]]

    system_prompts = [""] * len(prompts[0])

    responses = model.generate(system_prompts=system_prompts, input_texts=prompts)

    for i, question in enumerate(new_dicts):
        question["model_name"] = model_name_or_path
        question["metadata"]["rewrite_model"] = model.exact_model
        question["metadata"]["temperature"] = model.temperature
        del question["answers"]["answer2"]
        del question["level"]
        del question["prompt"]
        del question["model_id"]
        del question["model_name"]
        del question["temperature"]

        question["answers"]["answer1"]["answer"] = responses[i][0]

    write_file(src_file.replace(".json", file_ending), new_dicts)


def rewrite_onestopqa(
    model_name_or_path: str = "gpt-4.1",
    prompt_type: str = "simple",
    outfile: str = "data/readability_metrics/onestopqa_test_data/simple_prompt.json",
) -> None:
    from evaluation.calculate_readability_metrics import load_onestopqa

    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = get_model(
        model_name_or_path=model_name_or_path,
        config=config,
    )

    prompt_options = {
        "simple": simple_prompt,
        "complex": complex_prompt,
        "basic": basic_english_prompt,
    }
    prompt = prompt_options.get(prompt_type)
    if not prompt:
        raise ValueError("Prompt type must be one of [simple/complex/basic]")

    articles = load_onestopqa(reference=False)
    articles_truncated = [" ".join(paragraph[:1]) for paragraph in articles]
    prompts = [prompt(article) for article in articles_truncated]
    system_prompts = [""] * len(prompts)
    responses = model.generate(
        system_prompts=system_prompts,
        input_texts=prompts,
    )
    simple_language_paragraphs = [response[0] for response in responses]
    # write output list to file
    with open(outfile, "w") as file:
        json.dump(simple_language_paragraphs, file, indent=4)
    return


if __name__ == "__main__":
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="aave"
    )
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="simple"
    )
    rewrite_answers(
        src_file="data/generated_answers/gpt-4.1-answers.json", rewrite_type="errors"
    )
