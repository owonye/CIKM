from generation.normalize import exact_match, token_f1


def answer_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float | None]:
    if not gold_answers:
        return {"em": None, "f1": None}
    return {
        "em": max(exact_match(prediction, gold) for gold in gold_answers),
        "f1": max(token_f1(prediction, gold) for gold in gold_answers),
    }


def consistency_metrics(pre: float | None, post: float | None) -> dict[str, float | None]:
    if post is None:
        return {"consistency": None, "variance": None, "stability_gain": None}
    return {
        "consistency": post,
        "variance": 1.0 - post,
        "stability_gain": None if pre is None else post - pre,
    }
