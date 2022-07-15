import random
from num2words import num2words
import json

object_types = [
    ("pen", "pens"),
    ("cup", "cups"),
    ("spoon", "spoons"),
    ("towel", "towels"),
    ("book", "books"),
    ("flashlight", "flashlights"),
    ("key", "keys"),
    ("toothbrush", "toothbrushes"),
    ("flower", "flowers"),
    ("sock", "socks"),
    ("slipper", "slippers"),
    ("ball", "balls"),
]

colors = ["red", "blue", "green"]

n_types_bounds = (3, 6)
n_instances_bounds = (1, 8)


def get_types_counts_colors(type_indices):
    result = []
    for type_idx in type_indices:
        count = random.randint(*n_instances_bounds)
        color = random.choice(colors)
        type = object_types[type_idx][0] if count == 1 else object_types[type_idx][1]
        result.append((type, count, color))
    return result


def combine_noun_phrases(noun_phrases):
    if len(noun_phrases) == 1:
        return noun_phrases[0]
    if len(noun_phrases) == 2:
        return f"{noun_phrases[0]} and {noun_phrases[1]}"
    if len(noun_phrases) >= 3:
        return f"{', '.join(noun_phrases[:-1])}, and {noun_phrases[-1]}"
    raise ValueError()


def noun_phrase(object_type: str, count: int, color: str) -> str:
    count_word = num2words(count)
    return f"{count_word} {color} {object_type}"


def build_first_sentence(types_counts_colors):
    contents = combine_noun_phrases([noun_phrase(*tcc) for tcc in types_counts_colors])
    return f"I put {contents} into a box."


def build_second_sentence(removed_types) -> str:
    return f"Then I removed the {combine_noun_phrases(removed_types)}."


def build_question(color) -> str:
    return f"How many {color} items are left?"


def generate_example(object_types, colors, n_type_bounds, n_instance_bounds):
    # Add items to box
    n_types = random.randint(*n_types_bounds)
    type_indices = random.sample(range(len(object_types)), n_types)
    types_counts_colors = get_types_counts_colors(type_indices)
    first_sentence = build_first_sentence(types_counts_colors)

    # Remove items from box
    n_types_removed = random.randint(2, n_types - 1)
    selected_types = list(set(list(zip(*types_counts_colors))[0]))
    removed_types = random.sample(selected_types, n_types_removed)
    second_sentence = build_second_sentence(removed_types)

    # Pose the question
    remaining_types_counts_colors = [
        (type, count, color) for (type, count, color) in types_counts_colors if type not in removed_types
    ]
    remaining_colors = list(zip(*remaining_types_counts_colors))[2]
    color_in_question = random.choice(remaining_colors)
    question = build_question(color_in_question)

    # Find the answer
    answer_int = sum([count for (type, count, color) in remaining_types_counts_colors if color == color_in_question])
    targets = [answer_int, num2words(answer_int)]

    input_ = " ".join([first_sentence, second_sentence, question])
    return input_, targets


if __name__ == "__main__":
    examples = {"examples": []}
    for i in range(1000):
        input_, targets = generate_example(object_types, colors, n_types_bounds, n_instances_bounds)
        examples["examples"].append({"input": input_, "target": targets})

    with open("tasks/examples/set_tracking.json", "w") as f:
        json.dump(examples, f)
